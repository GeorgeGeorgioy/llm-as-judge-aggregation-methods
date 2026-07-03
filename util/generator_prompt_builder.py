from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_prompt_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load YAML prompt file."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dict.")

    if "messages" not in data or not isinstance(data["messages"], list):
        raise ValueError("YAML must contain a 'messages' list.")

    for i, m in enumerate(data["messages"]):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"Invalid message at index {i}. Each message must have role and content.")

    return data


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Load all rows from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        rows = list(reader)

    return rows


def build_prompt_object(
    dataset_name: str,
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
) -> Dict[str, Any]:
    if dataset_name == "HaluEval":
        return build_prompt_halueval(prompt_cfg, dataset_row)
    elif dataset_name == "BiasBio":
        return build_prompt_biasbio(prompt_cfg, dataset_row)
    elif dataset_name == "Arena":
        return build_prompt_arena(prompt_cfg, dataset_row)
    elif dataset_name == "ArenaPosition":
        return build_prompt_arenaposition(prompt_cfg, dataset_row)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_prompt_biasbio(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
) -> Dict[str, Any]:

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    user_filled = user_template.format(
        hard_text=dataset_row["hard_text"],
    )

    return {
        "id": str(dataset_row["id"]),
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["occupation"],
        "metadata": {
            "gender": dataset_row.get("gender"),
            "occupation": dataset_row.get("occupation"),
            "token_length": dataset_row.get("token_length"),
        },
    }


def build_prompt_halueval(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
) -> Dict[str, Any]:

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    user_filled = user_template.format(
        passage=dataset_row["passage"],
        question=dataset_row["question"],
        answer=dataset_row["answer"],
    )

    return {
        "id": str(dataset_row["id"]),
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["label"],
        "metadata": {
            "prompt_length": dataset_row.get("prompt_length"),
            "llama_3_1_bucket": dataset_row.get("llama_3_1_bucket"),
        },
    }


def build_prompt_arena(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
) -> Dict[str, Any]:

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    user_filled = user_template.format(
        prompt=dataset_row["prompt"],
        response_a=dataset_row["response_a"],
        response_b=dataset_row["response_b"],
    )

    return {
        "id": str(dataset_row["id"]),
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["winner"],
        "metadata": {
            "model_a": dataset_row.get("model_a"),
            "model_b": dataset_row.get("model_b"),
            "response_a_len": dataset_row.get("response_a_len"),
            "response_b_len": dataset_row.get("response_b_len"),
            "winner_model": dataset_row.get("winner_model"),
            "longer": dataset_row.get("longer"),
            "prompt_len": dataset_row.get("prompt_len"),
            "pair": dataset_row.get("pair"),
            "winner_A": dataset_row.get("winner_A"),
            "length_diff": dataset_row.get("length_diff"),
        },
    }


def build_prompt_arenaposition(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
) -> Dict[str, Any]:
    """
    Same as Arena, but response_a and response_b are swapped.
    """

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    user_filled = user_template.format(
        prompt=dataset_row["prompt"],
        response_a=dataset_row["response_b"],
        response_b=dataset_row["response_a"],
    )

    return {
        "id": str(dataset_row["id"]),
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["winner"],
        "metadata": {
            "model_a": dataset_row.get("model_a"),
            "model_b": dataset_row.get("model_b"),
            "response_a_len": dataset_row.get("response_a_len"),
            "response_b_len": dataset_row.get("response_b_len"),
            "winner_model": dataset_row.get("winner_model"),
            "longer": dataset_row.get("longer"),
            "prompt_len": dataset_row.get("prompt_len"),
            "pair": dataset_row.get("pair"),
            "winner_A": dataset_row.get("winner_A"),
            "length_diff": dataset_row.get("length_diff"),
            "position_swapped": True,
        },
    }


def build_and_save_jsonl(
    dataset_name: str,
    prompt_cfg: Dict[str, Any],
    dataset_rows: List[Dict[str, str]],
    out_path: Path,
) -> None:
    """Build generator JSONL and save it."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for dataset_row in dataset_rows:
            if "id" not in dataset_row:
                raise ValueError("Every dataset row must contain an 'id' field.")

            obj = build_prompt_object(
                dataset_name=dataset_name,
                prompt_cfg=prompt_cfg,
                dataset_row=dataset_row,
            )

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    config_path = Path("generator_jobs.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for job in config["jobs"]:
        dataset_name = job["dataset_name"]

        prompt_cfg = load_prompt_yaml(Path(job["yaml_path"]))
        dataset_rows = load_csv_rows(Path(job["dataset_path"]))

        build_and_save_jsonl(
            dataset_name=dataset_name,
            prompt_cfg=prompt_cfg,
            dataset_rows=dataset_rows,
            out_path=Path(job["output_path"]),
        )

        print("Dataset:", dataset_name)
        print("Loaded YAML keys:", list(prompt_cfg.keys()))
        print("Num messages:", len(prompt_cfg["messages"]))
        print("Dataset columns:", dataset_rows[0].keys() if dataset_rows else "NO ROWS")
        print("Num dataset rows:", len(dataset_rows))
        print("Saved generator prompts to:", Path(job["output_path"]))
        print("-" * 80)