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


def load_jsonl_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load all rows from JSONL."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rows: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {jsonl_path}: {e}")

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {jsonl_path} is not a JSON object.")

            rows.append(obj)

    return rows


def index_rows_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index rows by id."""
    indexed: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if "id" not in row:
            raise ValueError("Every row must contain an 'id' field.")

        row_id = str(row["id"])

        if row_id in indexed:
            raise ValueError(f"Duplicate id found: {row_id}")

        indexed[row_id] = row

    return indexed

###### -- HELPER FUNCTIONS -- ######

def clean_prediction(raw_prediction: Any) -> str:
    """Keep only the first valid label from the generator prediction."""
    fallback = "The generator did not provide a valid answer. Please give your own judgment."

    if raw_prediction is None:
        return fallback

    text = str(raw_prediction).strip()
    if not text:
        return fallback

    first_word = text.split()[0].strip().upper()

    allowed = {
    "PASS", "FAIL", "A", "B", "PROPOSED_ANSWER", "ANSWER",
    "PHOTOGRAPHER",
    "FILMMAKER",
    "PHYSICIAN",
    "COMPOSER",
    "COMEDIAN",
    "ARCHITECT",
    "PERSONAL_TRAINER",
    "ATTORNEY",
    "INTERIOR_DESIGNER",
    "PROFESSOR",
    "PASTOR",
    "SOFTWARE_ENGINEER",
    "PAINTER",
    "YOGA_TEACHER",
    "DIETITIAN",
    "PARALEGAL",
    "CHIROPRACTOR",
    "DENTIST",
    "RAPPER",
    "TEACHER",
    "PSYCHOLOGIST",
    "ACCOUNTANT",
    "NURSE",
    "DJ",
    "MODEL",
    "JOURNALIST",
    "POET",
    "SURGEON"
}
    

    if first_word in allowed:
        
        return first_word

    return fallback    


#### -- END OF HEALPER FUNCTIONS -- ####

def build_prompt_object(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
    generator_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one judge prompt object."""

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    sample_id = str(dataset_row["id"])

    if "prediction" not in generator_row:
        raise ValueError(f"Missing 'prediction' in generator output for id={sample_id}")

    #proposed_answer = generator_row["prediction"]
    raw_prediction = generator_row["prediction"]
    proposed_answer = clean_prediction(raw_prediction)

    #---------- delete here ----------

    passage = dataset_row["passage"]
    question = dataset_row["question"]
    answer = dataset_row["answer"]

    user_filled = user_template.format(
        passage=passage,
        question=question,
        answer=answer,
        proposed_answer=proposed_answer,
    )

    obj = {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["label"],
        "metadata": {
            "generators_answer": proposed_answer,
            "prompt_length": dataset_row.get("prompt_length"),
            "llama_3_1_bucket": dataset_row.get("llama_3_1_bucket"),

        }
    }

    return obj


def build_and_save_jsonl(
    prompt_cfg: Dict[str, Any],
    dataset_rows: List[Dict[str, str]],
    generator_rows: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Build judge JSONL and save it."""

    generator_by_id = index_rows_by_id(generator_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for dataset_row in dataset_rows:
            if "id" not in dataset_row:
                raise ValueError("Every dataset row must contain an 'id' field.")

            sample_id = str(dataset_row["id"])

            if sample_id not in generator_by_id:
                raise ValueError(f"Missing generator output for id={sample_id}")

            generator_row = generator_by_id[sample_id]

            obj = build_prompt_object(prompt_cfg, dataset_row, generator_row)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":



    """
    Name of output judge prompt folder gen_<model_name>_<aggre.method>_for_<dataset_name>_judge.jsonl
    Name of input generator results folder generator_<model>_<method>_<dataset_name>_results.jsonl
    """

 ################ -- arena -- ########################################################
    """
    yaml_path = Path("/work3/s233559/Thesis/data/templates/ArenaPosition_j.yaml")
    dataset_path = Path("/work3/s233559/Thesis/data/dataset/Chatbot_arena_2000_final.csv")
    generator_output_path = Path("/work3/s233559/Thesis/results/generator/llama8/generator_llama8_oneshot_ArenaPosition_results.jsonl")
    output_path = Path("/work3/s233559/Thesis/prompts/judge/generator_llama8_oneshot_ArenaPosition_to_judge.jsonl")
    """
 ############## -- biasandbio -- ########################################################
    """
    yaml_path = Path("/work3/s233559/Thesis/data/templates/BiasBio_j.yaml")
    dataset_path = Path("/work3/s233559/Thesis/data/dataset/bias_in_bios_2000_final.csv")
    generator_output_path = Path("/work3/s233559/Thesis/results/generator/qwen7/generator_qwen7_oneshot_BiasBio_results.jsonl")
    output_path = Path("/work3/s233559/Thesis/prompts/judge/generator_qwen7_oneshot_BiasBio_to_judge.jsonl")
    """

    #####################-- helueval --########################################
    
    yaml_path = Path("/work3/s233559/Thesis/data/templates/HaluEval_j.yaml")
    dataset_path = Path("/work3/s233559/Thesis/data/dataset/hallucination_eval_2000_balanced_clean.csv")
    generator_output_path = Path("/work3/s233559/Thesis/results/generator/qwen7/generator_qwen7_oneshot_HaluEval_results.jsonl")
    output_path = Path("/work3/s233559/Thesis/prompts/judge/generator_qwen7_oneshot_HaluEval_to_judge.jsonl")
 
    
    
    ########################################################################
    prompt_cfg = load_prompt_yaml(yaml_path)
    dataset_rows = load_csv_rows(dataset_path)
    generator_rows = load_jsonl_rows(generator_output_path)

    build_and_save_jsonl(prompt_cfg, dataset_rows, generator_rows, output_path)

    print("Loaded YAML keys:", list(prompt_cfg.keys()))
    print("Num messages:", len(prompt_cfg["messages"]))
    print("Dataset columns:", dataset_rows[0].keys() if dataset_rows else "NO ROWS")
    print("Num dataset rows:", len(dataset_rows))
    print("Num generator rows:", len(generator_rows))
    print("Saved judge prompts to:", output_path)


    """
      hallueval
      
    passage = dataset_row["passage"]
    question = dataset_row["question"]
    answer = dataset_row["answer"]

    user_filled = user_template.format(
        passage=passage,
        question=question,
        answer=answer,
        proposed_answer=proposed_answer,
    )

    obj = {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["label"],
        "metadata": {
            "generators_answer": proposed_answer,
            "prompt_length": dataset_row.get("prompt_length"),
            "llama_3_1_bucket": dataset_row.get("llama_3_1_bucket"),

        }
    }

    return obj


    ===============================================================
     
    biasandbio

    passage = dataset_row["hard_text"]
    #question = "What is the occupation of the person described in the passage?"
    answer = dataset_row["occupation"]

    user_filled = user_template.format(
        hard_text=passage,
        #question=question,
        #answer=answer,
        proposed_occupation=proposed_answer,
    )

    obj = {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["occupation"],
        "metadata": {
            "generators_answer": proposed_answer,
            "gender": dataset_row.get("gender"),
            "occupation": dataset_row.get("occupation"),
            "token_length": dataset_row.get("token_length"),
        }
    }

    return obj
    

    ================================================================

 prompt = dataset_row["prompt"]
    response_a = dataset_row["response_a"]
    response_b = dataset_row["response_b"]

    user_filled = user_template.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        proposed_answer=proposed_answer,
    )

    obj = {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["winner"],
        "metadata": {
            "generators_answer": proposed_answer,
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
        }
    }


    return obj

    """