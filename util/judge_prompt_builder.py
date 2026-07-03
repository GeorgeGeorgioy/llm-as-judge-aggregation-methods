from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml
import re
import pandas as pd



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

def normalize(x):
    if pd.isna(x):
        return ""
    
    x = str(x)
    
    # lower
    x = x.lower()
    # replace backslashes with space
    x = x.replace("\\", " ")

    
    # strip leading/trailing spaces
    x = x.strip()
    
    # remove newlines / tabs
    x = re.sub(r"[\n\r\t]", " ", x)
    
    # collapse multiple spaces
    x = re.sub(r"\s+", " ", x)
    # convert spaces to underscore
    x = x.replace(" ", "_")
    # collapse multiple underscores into one
    x = re.sub(r"_+", "_", x)
    
    # remove surrounding punctuation
    x = x.strip(".,:;!?\"'()[]{}")
    return x

ALLOWED_OCCUPATIONS = [
    "photographer",
    "filmmaker",
    "physician",
    "composer",
    "comedian",
    "architect",
    "personal_trainer",
    "attorney",
    "interior_designer",
    "professor",
    "pastor",
    "software_engineer",
    "painter",
    "yoga_teacher",
    "dietitian",
    "paralegal",
    "chiropractor",
    "dentist",
    "rapper",
    "teacher",
    "psychologist",
    "accountant",
    "nurse",
    "dj",
    "model",
    "journalist",
    "poet",
    "surgeon",
]

ALLOWED_OCCUPATIONS_NORM = [normalize(x) for x in ALLOWED_OCCUPATIONS]

def extract_valid_occupation(x):
    if pd.isna(x):
        return "invalid"
    
    x = normalize(x)

    found = []

    for occupation in ALLOWED_OCCUPATIONS_NORM:
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(occupation)}(?![a-zA-Z0-9_])"

        if re.search(pattern, x):
            found.append(occupation)

    if len(found) == 1:
        return found[0]
    
    return "invalid"

def normalize_arena(x: Any) -> str:
    """
    Normalize raw model output into A / B.
    """

    if x is None:
        return "invalid"

    text = str(x).strip().upper()

    if not text:
        return "invalid"

    #  standalone A or B 
    match = re.search(r"\b(A|B)\b", text)

    if match:
        return match.group(1)

    return "invalid"




def clean_prediction(dataset_name: str, raw_prediction: Any) -> str:
    """Clean generator prediction based on dataset name."""
    fallback = "invalid"

    if raw_prediction is None:
        cleaned = fallback

    else:
        text = str(raw_prediction).strip()

        if not text:
            cleaned = fallback

        else:
            cleaned = fallback  # default

            if dataset_name == "HaluEval":
                normalized_text = text.lower()

                match = re.search(
                    r"\b(pass|passes|passed|passing|fail|fails|failed|failing)\b",
                    normalized_text
                )

                if match:
                    token = match.group(1)

                    if token in ["pass", "passes", "passed", "passing"]:
                        cleaned = "PASS"

                    if token in ["fail", "fails", "failed", "failing"]:
                        cleaned = "FAIL"

            if dataset_name in {"Arena", "ArenaPosition"}:
                normalized_text = text.upper()

                match = re.search(r"\b(A|B)\b", normalized_text)

                if match:
                    cleaned = match.group(1)

            # ---------------- BiasBio ----------------
            if dataset_name == "BiasBio":
                cleaned = extract_valid_occupation(raw_prediction)                



    raw_for_log = str(raw_prediction).replace("\n", "\\n").replace("\r", "\\r")
    with open("cleaning_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{dataset_name} | {raw_for_log} | {cleaned}\n")

    return cleaned




#### -- END OF HEALPER FUNCTIONS -- ####



def build_prompt_object(dataset_name, prompt_cfg, dataset_row, generator_row):
    if dataset_name == "HaluEval":
        return build_prompt_halueval(prompt_cfg, dataset_row, generator_row, dataset_name)
    elif dataset_name == "BiasBio":
        return build_prompt_biasbio(prompt_cfg, dataset_row, generator_row, dataset_name)
    elif dataset_name == "Arena":
        return build_prompt_arena(prompt_cfg, dataset_row, generator_row, dataset_name)
    elif dataset_name == "ArenaPosition":
        return build_prompt_arenaposition(prompt_cfg, dataset_row, generator_row, dataset_name)    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def build_prompt_biasbio(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
    generator_row: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """Build one judge prompt object."""

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    sample_id = str(dataset_row["id"])

    if "prediction" not in generator_row:
        raise ValueError(f"Missing 'prediction' in generator output for id={sample_id}")

    #proposed_answer = generator_row["prediction"]
    raw_prediction = generator_row["prediction"]
    proposed_answer = clean_prediction(dataset_name, raw_prediction)

    user_filled = user_template.format(
        hard_text=dataset_row["hard_text"],
        proposed_occupation=proposed_answer,
    )

    return {
        "id": str(dataset_row["id"]),
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
        },
    }

def  build_prompt_arenaposition(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
    generator_row: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """ArenaPosition.csv is the same as Arena.csv 
       I swap only the responses
    """

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    sample_id = str(dataset_row["id"])

    if "prediction" not in generator_row:
        raise ValueError(f"Missing 'prediction' in generator output for id={sample_id}")

    #proposed_answer = generator_row["prediction"]
    raw_prediction = generator_row["prediction"]
    proposed_answer = clean_prediction(dataset_name, raw_prediction)
    """ArenaPosition.csv is the same as Arena.csv """

    user_filled = user_template.format(
        prompt=dataset_row["prompt"],
        response_a=dataset_row["response_b"],
        response_b=dataset_row["response_a"],
        proposed_answer=proposed_answer,
    )

    return {
        "id": str(dataset_row["id"]),
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
        },
    }





def  build_prompt_arena(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
    generator_row: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """Build one judge prompt object."""

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    sample_id = str(dataset_row["id"])

    if "prediction" not in generator_row:
        raise ValueError(f"Missing 'prediction' in generator output for id={sample_id}")

    #proposed_answer = generator_row["prediction"]
    raw_prediction = generator_row["prediction"]
    proposed_answer = clean_prediction(dataset_name, raw_prediction)

    user_filled = user_template.format(
        prompt=dataset_row["prompt"],
        response_a=dataset_row["response_a"],
        response_b=dataset_row["response_b"],
        proposed_answer=proposed_answer,
    )

    return {
        "id": str(dataset_row["id"]),
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
        },
    }






def build_prompt_halueval(
    prompt_cfg: Dict[str, Any],
    dataset_row: Dict[str, str],
    generator_row: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """Build one judge prompt object."""

    system_template = prompt_cfg["messages"][0]["content"]
    user_template = prompt_cfg["messages"][1]["content"]

    sample_id = str(dataset_row["id"])

    if "prediction" not in generator_row:
        raise ValueError(f"Missing 'prediction' in generator output for id={sample_id}")

    #proposed_answer = generator_row["prediction"]
    raw_prediction = generator_row["prediction"]
    proposed_answer = clean_prediction(dataset_name, raw_prediction)

    #---------- delete here ----------

    user_filled = user_template.format(
        passage=dataset_row["passage"],
        question=dataset_row["question"],
        answer=dataset_row["answer"],
        proposed_answer=proposed_answer,
    )

    return {
        "id": str(dataset_row["id"]),
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_filled},
        ],
        "ground_truth": dataset_row["label"],
        "metadata": {
            "generators_answer": proposed_answer,
            "prompt_length": dataset_row.get("prompt_length"),
            "llama_3_1_bucket": dataset_row.get("llama_3_1_bucket"),
        },
    }


def build_and_save_jsonl(
    dataset_name: str,
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
                print(f"[WARNING] Missing generator output for id={sample_id} - skipping")
                #raise ValueError(f"Missing generator output for id={sample_id}")
                continue
                
                

            generator_row = generator_by_id[sample_id]

            obj = build_prompt_object(dataset_name=dataset_name, prompt_cfg=prompt_cfg, dataset_row=dataset_row, generator_row=generator_row,)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    
    with open("cleaning_log.txt", "w", encoding="utf-8") as f:
     f.write("DATASET | RAW PREDICTION | CLEANED PREDICTION\n")
     f.write("-" * 60 + "\n")



    """
    Name of output judge prompt folder gen_<model_name>_<aggre.method>_for_<dataset_name>_judge.jsonl
    Name of input generator results folder generator_<model>_<method>_<dataset_name>_results.jsonl
    """

    config_path = Path("judge_jobs.yaml")

    config = yaml.safe_load(config_path.read_text())

    for job in config["jobs"]:
        dataset_name = job["dataset_name"]
        prompt_cfg = load_prompt_yaml(Path(job["yaml_path"]))
        dataset_rows = load_csv_rows(Path(job["dataset_path"]))
        generator_rows = load_jsonl_rows(Path(job["generator_output_path"]))

        build_and_save_jsonl(
            dataset_name=dataset_name,
            prompt_cfg=prompt_cfg,
            dataset_rows=dataset_rows,
            generator_rows=generator_rows,
            out_path=Path(job["output_path"]),
        )

        print("Loaded YAML keys:", list(prompt_cfg.keys()))
        print("Num messages:", len(prompt_cfg["messages"]))
        print("Dataset columns:", dataset_rows[0].keys() if dataset_rows else "NO ROWS")
        print("Num dataset rows:", len(dataset_rows))
        print("Num generator rows:", len(generator_rows))
        print("Saved judge prompts to:", Path(job["output_path"]))




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