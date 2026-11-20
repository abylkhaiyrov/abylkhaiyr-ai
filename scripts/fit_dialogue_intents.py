from __future__ import annotations

import json
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.conversation import CASES
from llm.reasoning.intent import classify_intent
from llm.reasoning.intent_model import IntentModel

LABELS = {
    "greeting": "dialogue",
    "general_fact": "information",
    "explain_concept": "information",
    "clarification": "clarify",
    "multi_turn_name": "multi_turn",
    "multi_turn_identity": "multi_turn",
    "context_qa": "context",
    "translation": "translation",
    "multi_turn_translation": "translation",
    "multi_turn_math": "math",
    "list_items": "instruction",
    "text_analysis": "analysis",
    "explain_code_concept": "code",
}
EVALUATION_LABELS = {
    "greeting": "dialogue",
    "identity": "dialogue",
    "explain": "information",
    "clarify": "clarify",
    "multi_turn": "multi_turn",
    "context": "context",
    "translation": "translation",
    "math": "math",
    "instruction": "instruction",
    "code": "code",
}


def main() -> None:
    examples = []
    for source in Path("data/dialogue_v4/sft_groups").glob("train.*.jsonl"):
        for row in read_jsonl(source):
            category = row.get("category", "")
            label = LABELS.get(category)
            if category.startswith("identity_"):
                label = "dialogue"
            if category.startswith("reading_"):
                label = "analysis"
            if label:
                prompts = [m["content"] for m in row["messages"] if m["role"] == "user"]
                examples.append((prompts[-1], label))
    model = IntentModel.fit(examples)
    cases = [item for item in CASES if item.category in EVALUATION_LABELS]
    assessed = []
    for item in cases:
        prediction = model.predict(item.prompt)
        assessed.append(
            {
                "id": item.id,
                "label": EVALUATION_LABELS[item.category],
                "heuristic": classify_intent(item.prompt),
                "prediction": prediction.label,
                "similarity": prediction.similarity,
                "margin": prediction.margin,
            }
        )
    result = {
        "count": len(cases),
        "learned_correct": sum(row["label"] == row["prediction"] for row in assessed),
        "heuristic_correct": sum(row["label"] == row["heuristic"] for row in assessed),
        "abstained": sum(row["prediction"] is None for row in assessed),
    }
    output = Path("experiments/dialogue_v4")
    output.mkdir(parents=True, exist_ok=True)
    (output / "intent_model.json").write_text(model.model_dump_json(), encoding="utf-8")
    (output / "intent_report.json").write_text(
        json.dumps({"metrics": result, "cases": assessed}, indent=2), encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()
