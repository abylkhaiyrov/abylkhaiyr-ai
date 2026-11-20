from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional

from llm.evaluation.comparison import load_engine
from llm.tokenizer.bpe import Message
from llm.training.losses import IGNORE_INDEX, causal_lm_loss_sum

PROBES = [
    "Как дела?",
    "Привет, как настроение?",
    "Что ты умеешь делать?",
    "Что такое искусственный интеллект?",
    "Почему идёт дождь?",
    "Как дела?",
    "How are you?",
    "What is the difference between a cat and a dog?",
    "Сәлем, қалайсың?",
    "Жаңбыр неге жауады?",
]


def load_rows(path: Path, limit: int) -> list[list[Message]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append([Message(m["role"], m["content"]) for m in json.loads(line)["messages"]])
            if len(rows) == limit:
                break
    return rows


def probe_distribution(engine, prompt: str) -> dict:
    tokenizer = engine.tokenizer
    ids = tokenizer.encode_conversation([Message("user", prompt)], add_generation_prompt=True)
    prompt_set = set(ids[2:-2])
    copy_masses, eos_probs, entropies = [], [], []
    history = list(ids)
    first_top: list[tuple[str, float]] = []
    with torch.inference_mode():
        for step in range(48):
            logits = engine.model(torch.tensor([history], device=engine.device))[0, -1].float()
            probs = functional.softmax(logits, dim=-1).cpu()
            if step == 0:
                top = probs.topk(5)
                first_top = [
                    (tokenizer.decode([i]), round(float(p), 4))
                    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
                ]
            copy_masses.append(float(probs[list(prompt_set)].sum()) if prompt_set else 0.0)
            eos_probs.append(float(probs[tokenizer.eos_id]))
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
            token = int(probs.argmax())
            if token == tokenizer.eos_id:
                break
            history.append(token)
    generated = history[len(ids) :]
    echoed = sum(1 for t in generated if t in prompt_set) / max(len(generated), 1)
    return {
        "prompt": prompt,
        "first_token_top5": first_top,
        "mean_copy_mass": round(statistics.fmean(copy_masses), 4),
        "max_eos_prob": round(max(eos_probs), 4),
        "mean_entropy_nats": round(statistics.fmean(entropies), 3),
        "generated_token_prompt_overlap": round(echoed, 3),
        "answer": tokenizer.decode(generated, skip_special_tokens=True),
    }


def teacher_forced(engine, rows: list[list[Message]]) -> dict:
    tokenizer = engine.tokenizer
    total_loss, total_tokens, copy_hits, positions = 0.0, 0, 0, 0
    eos_gold_probs: list[float] = []
    with torch.inference_mode():
        for messages in rows:
            ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
            ids, labels = ids[:511], labels[:511]
            inputs = torch.tensor([ids[:-1]], device=engine.device)
            targets = torch.tensor([labels[1:]], device=engine.device)
            logits = engine.model(inputs)
            loss, count = causal_lm_loss_sum(logits, targets)
            total_loss += float(loss)
            total_tokens += count
            probs = functional.softmax(logits[0].float(), dim=-1)
            user_tokens = set(ids[: next(i for i, t in enumerate(ids) if t == tokenizer.eos_id)])
            for index, target in enumerate(targets[0].tolist()):
                if target == IGNORE_INDEX:
                    continue
                positions += 1
                copy_hits += int(int(probs[index].argmax()) in user_tokens)
                if target == tokenizer.eos_id:
                    eos_gold_probs.append(float(probs[index, target]))
    return {
        "assistant_loss": round(total_loss / max(total_tokens, 1), 4),
        "argmax_in_user_tokens_rate": round(copy_hits / max(positions, 1), 4),
        "gold_eos_prob_mean": round(statistics.fmean(eos_gold_probs), 4) if eos_gold_probs else 0,
    }


def gradient_flow(engine, rows: list[list[Message]]) -> dict[str, float]:
    tokenizer = engine.tokenizer
    model = engine.model
    model.zero_grad()
    for messages in rows:
        ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
        ids, labels = ids[:511], labels[:511]
        logits = model(torch.tensor([ids[:-1]], device=engine.device))
        loss, count = causal_lm_loss_sum(logits, torch.tensor([labels[1:]], device=engine.device))
        (loss / max(count, 1) / len(rows)).backward()
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = ".".join(name.split(".")[:2]) if name.startswith("blocks") else name.split(".")[0]
        norms[key] = norms.get(key, 0.0) + float(parameter.grad.pow(2).sum())
    model.zero_grad()
    return {key: round(value**0.5, 5) for key, value in norms.items()}


def representation_health(engine, rows: list[list[Message]]) -> dict:
    tokenizer = engine.tokenizer
    hidden: list[torch.Tensor] = []
    handle = engine.model.final_norm.register_forward_hook(
        lambda _m, _i, out: hidden.append(out[0].detach().float().cpu())
    )
    with torch.inference_mode():
        for messages in rows:
            ids = tokenizer.encode_conversation(messages)[:511]
            engine.model(torch.tensor([ids], device=engine.device))
    handle.remove()
    states = torch.cat(hidden)
    normed = functional.normalize(states, dim=-1)
    sample = normed[torch.randperm(len(normed))[:512]]
    cosine = sample @ sample.T
    off_diagonal = (cosine.sum() - cosine.diagonal().sum()) / (len(sample) * (len(sample) - 1))
    singular = torch.linalg.svdvals(states - states.mean(0))
    spectrum = singular.pow(2) / singular.pow(2).sum()
    effective_rank = float(torch.exp(-(spectrum * spectrum.clamp_min(1e-12).log()).sum()))
    embedding = functional.normalize(engine.model.embedding.weight.detach().float().cpu(), dim=-1)
    pick = embedding[torch.randperm(len(embedding))[:1024]]
    emb_cos = pick @ pick.T
    emb_off = (emb_cos.sum() - emb_cos.diagonal().sum()) / (len(pick) * (len(pick) - 1))
    return {
        "hidden_mean_pairwise_cosine": round(float(off_diagonal), 4),
        "hidden_effective_rank": round(effective_rank, 1),
        "hidden_dim": states.shape[1],
        "embedding_mean_pairwise_cosine": round(float(emb_off), 4),
    }


def weight_stats(engine) -> dict[str, dict[str, float]]:
    stats = {}
    for name, parameter in engine.model.named_parameters():
        if parameter.ndim >= 2 and name.startswith("blocks"):
            data = parameter.detach().float()
            stats[name] = {
                "std": round(float(data.std()), 4),
                "absmax": round(float(data.abs().max()), 3),
            }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(prog="diagnose_network")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--data", default="data/v3_3/sft_groups/validation.dialog.jsonl")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    rows = load_rows(Path(args.data), args.rows)
    report = {
        "checkpoint": args.checkpoint,
        "probes": [probe_distribution(engine, prompt) for prompt in dict.fromkeys(PROBES)],
        "teacher_forced": teacher_forced(engine, rows),
        "gradient_norm_by_layer": gradient_flow(engine, rows[:16]),
        "representation": representation_health(engine, rows),
        "weights": weight_stats(engine),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("weights", "probes")}
    print(json.dumps(summary, indent=1))
    for probe in report["probes"]:
        print(json.dumps(probe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
