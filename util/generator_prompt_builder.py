from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List
import json
import yaml


def load_prompt_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load YAML prompt registry entry (single prompt file)."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")



    if "messages" not in data or not isinstance(data["messages"], list):
        raise ValueError("YAML must contain a 'messages' list.")

    # basic validation for messages
    for i, m in enumerate(data["messages"]):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"Invalid message at index {i}. Each must have role+content.")

    # optional: variables list
    if "variables" in data and not isinstance(data["variables"], list):
        raise ValueError("'variables' must be a list if present.")

    return data


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]: # not memory efficient for large CSVs, but fine for small datasets
    """Load all rows from a CSV into list of dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row (fieldnames).")

        rows = list(reader)

    return rows



def build_prompt_object(prompt_cfg: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
    """Build prompt json by filling in variables from a CSV row."""

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    prompt = row["prompt"]
    response_a = row["response_a"]
    response_b = row["response_b"]

    sample_id = row["id"]
    label = row["winner"]

    user_filled = user_template.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b
    )

    obj = {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": label,
        "metadata": {
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "response_a_len": row["response_a_len"],
            "response_b_len": row["response_b_len"],
            "winner_model": row["winner_model"],
            "longer": row["longer"],
            "prompt_len": row["prompt_len"],
            "pair": row["pair"],
            "winner_A": row["winner_A"],
            "length_diff": row["length_diff"],
            "swapped": True
        }
    }

    return obj

    


def build_and_save_jsonl( prompt_cfg: Dict[str, Any], rows: List[Dict[str, str]], out_path: Path) -> None:

      with out_path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            obj = build_prompt_object(prompt_cfg, row)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")




if __name__ == "__main__":
    yaml_path = Path("/work3/s233559/Thesis/data/templates/arena_g_swaped.yalm")
    csv_path = Path("/work3/s233559/Thesis/data/dataset/Chatbot_arena_2000_final.csv")
    output_path = Path("/work3/s233559/Thesis/prompts/generator/generator_Chatbot_arena_swapped_v1.jsonl")

    prompt_cfg = load_prompt_yaml(yaml_path)
    rows = load_csv_rows(csv_path)
    build_and_save_jsonl(prompt_cfg, rows, output_path)

    print("Loaded YAML keys:", list(prompt_cfg.keys()))
    print("Num messages:", len(prompt_cfg["messages"]))
    print("CSV columns:", rows[0].keys() if rows else "NO ROWS")
    print("Num rows:", len(rows))