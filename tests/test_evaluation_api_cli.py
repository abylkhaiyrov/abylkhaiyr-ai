import json
import math

import pytest
import torch
from fastapi.testclient import TestClient

from llm.api.server import DEFAULT_HOST, create_app
from llm.cli.main import build_parser, main
from llm.core.errors import ConfigError, HardwareError
from llm.core.hardware import detect_hardware, resolve_device, resolve_dtype
from llm.core.profiles import available_profiles, estimate_training_memory_gb, fits_hardware
from llm.evaluation import suites
from llm.evaluation.metrics import exact_match, pass_at_k, repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.training.losses import dpo_loss, sequence_log_probs


def test_pass_at_k_known_values():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 4) == 1.0
    assert pass_at_k(5, 2, 2) == pytest.approx(1 - math.comb(3, 2) / math.comb(5, 2))


def test_text_metrics():
    assert exact_match(" Paris. ", "paris")
    assert repeated_ngram_ratio([1, 2, 3] * 10) > 0.8
    assert repeated_ngram_ratio(list(range(30))) == 0.0


def test_generated_arithmetic_references_are_correct_and_reproducible():
    cases = suites.arithmetic_cases(20, seed=1)
    assert cases == suites.arithmetic_cases(20, seed=1)
    for case in cases:
        expression = case.prompt.removeprefix("What is ").removesuffix("?")
        assert int(case.reference) == eval(expression)


def test_constraint_checkers():
    cases = {c.constraint + c.argument: c for c in suites.instruction_cases()}
    assert suites.check_constraint(cases["containsapple"], "An Apple a day")
    assert not suites.check_constraint(cases["lowercase"], "Hello")
    assert suites.check_constraint(cases["ends_withdone"], "All finished, done.")
    assert not suites.check_constraint(cases["max_words5"], "one two three four five six")


def test_dpo_loss_is_ln2_at_reference_and_decreases_with_preference():
    value = torch.tensor([-10.0])
    assert dpo_loss(value, value, value, value, 0.1).item() == pytest.approx(math.log(2))
    better = dpo_loss(value + 5, value - 5, value, value, 0.1)
    assert better.item() < math.log(2)


def test_sequence_log_probs_ignores_masked_targets():
    logits = torch.zeros(1, 3, 4)
    targets = torch.tensor([[1, -100, 2]])
    assert sequence_log_probs(logits, targets).item() == pytest.approx(2 * math.log(0.25))


def test_sequence_log_probs_ignore_impossible_token_at_masked_target():
    logits = torch.tensor([[[float("-inf"), 0.0], [0.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[-100, 1]])
    score = sequence_log_probs(logits, targets)
    assert score.item() == pytest.approx(-math.log(2))
    score.backward()
    assert logits.grad.isfinite().all()
    assert logits.grad[0, 0].count_nonzero() == 0


@pytest.fixture()
def client(model_config, tokenizer):
    torch.manual_seed(0)
    engine = InferenceEngine(CausalLM(model_config), tokenizer, torch.device("cpu"))
    return TestClient(create_app(engine, "test-model"))


def test_api_models_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"


def test_api_chat_completion_shape(client):
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 1,
    }
    response = client.post("/v1/chat/completions", json=body).json()
    assert response["object"] == "chat.completion"
    choice = response["choices"][0]
    assert choice["message"]["role"] == "assistant" and choice["finish_reason"] in {
        "stop",
        "length",
    }
    usage = response["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert client.post("/v1/chat/completions", json=body).json()["choices"] == response["choices"]


def test_api_usage_reports_fitted_prompt_tokens(model_config, tokenizer, tmp_path):
    engine = InferenceEngine(CausalLM(model_config), tokenizer, torch.device("cpu"))
    client = TestClient(create_app(engine, "test-model", studio_root=tmp_path))
    body = {
        "messages": [{"role": "user", "content": "a" * 1000}],
        "max_tokens": 8,
        "temperature": 0,
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == engine.max_positions - 8


def test_api_streaming_matches_non_streaming(client):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12, "temperature": 0}
    full = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
    streamed = ""
    lines = client.post("/v1/chat/completions", json={**body, "stream": True}).text.split("\n\n")
    assert lines[-2] == "data: [DONE]"
    for line in lines[:-2]:
        streamed += json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get(
            "content", ""
        )
    assert streamed == full


def test_api_validation_errors(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    bad_role = {"messages": [{"role": "hacker", "content": "x"}]}
    assert client.post("/v1/chat/completions", json=bad_role).status_code == 422
    two = {"messages": [{"role": "user", "content": "x"}], "n": 2}
    assert client.post("/v1/chat/completions", json=two).status_code == 400


def test_api_is_local_by_default():
    assert DEFAULT_HOST == "127.0.0.1"
    assert build_parser().parse_args(["serve", "--checkpoint", "x"]).host == "127.0.0.1"


def test_cli_reports_errors_with_exit_code(capsys, tmp_path):
    code = main(["train", "--config", str(tmp_path / "missing.yaml")])
    assert code == ConfigError.exit_code
    assert "config file not found" in capsys.readouterr().err


def test_cli_exposes_all_required_commands():
    parser = build_parser()
    for command in [
        "hardware",
        "prepare-data",
        "train-tokenizer",
        "train",
        "resume",
        "finetune",
        "evaluate",
        "generate",
        "chat",
        "serve",
        "studio",
        "benchmark",
    ]:
        assert (
            parser.parse_args(
                [
                    command,
                    *(
                        ["--checkpoint", "c"]
                        if command in {"evaluate", "chat", "serve", "studio", "benchmark"}
                        else []
                    ),
                    *(["p"] if command == "generate" else []),
                    *(["--checkpoint", "c"] if command == "generate" else []),
                ]
            ).command
            == command
        )


def test_hardware_detection_and_profiles():
    info = detect_hardware()
    assert info.cpu_cores >= 1 and info.ram_gb > 0 and info.device in {"cuda", "mps", "cpu"}
    runnable = available_profiles(8192, info)
    assert "debug" in runnable
    assert not fits_hardware("large", 8192, info) or info.device_memory_gb > 30


def test_memory_estimate_grows_with_batch(model_config):
    assert estimate_training_memory_gb(model_config, 8, 64) > estimate_training_memory_gb(
        model_config, 1, 64
    )


def test_device_and_precision_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32
    with pytest.raises(HardwareError):
        resolve_dtype("fp16", torch.device("cpu"))
    if not torch.cuda.is_available():
        with pytest.raises(HardwareError):
            resolve_device("cuda")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_is_finite_on_available_devices(model_config, dtype):
    device = resolve_device("auto")
    if dtype == torch.bfloat16 and device.type == "mps":
        pytest.skip("bf16 autocast is not exercised on MPS")
    model = CausalLM(model_config).to(device).eval()
    ids = torch.randint(0, model_config.vocab_size, (2, 16), device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits = model(ids)
    assert torch.isfinite(logits.float()).all()


def test_cpu_and_default_device_agree(model_config):
    device = resolve_device("auto")
    cpu_model = CausalLM(model_config).eval()
    other = CausalLM(model_config).to(device).eval()
    other.load_state_dict(cpu_model.state_dict())
    ids = torch.randint(0, model_config.vocab_size, (1, 10))
    with torch.no_grad():
        assert torch.allclose(cpu_model(ids), other(ids.to(device)).cpu(), atol=1e-3)
