from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return items


def messages_to_text(messages: List[Dict[str, str]], mode: str = "plain") -> str:
    """
    Convert chat messages to a single text blob for token counting.

    mode="plain": simple concatenation with role headers (stable & tokenizer-agnostic)
    """
    if mode != "plain":
        raise ValueError("Only mode='plain' is supported in this script.")

    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}\n")
    return "\n".join(parts).strip() + "\n"


def load_hf_tokenizer(model_id: str):
    """
    Load a Hugging Face tokenizer (best choice for Qwen2.5-14B).
    """
    from transformers import AutoTokenizer  # pip install transformers

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    return tok


def count_tokens_hf(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_tiktoken(encoding: str = "o200k_base"):
    """
    Optional: load tiktoken encoding (rough baseline; not Qwen-accurate).
    """
    import tiktoken  # pip install tiktoken

    return tiktoken.get_encoding(encoding)


def count_tokens_tiktoken(enc, text: str) -> int:
    return len(enc.encode(text))


def summarize(lengths: List[int]) -> Dict[str, float]:
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    if n == 0:
        return {}

    def percentile(p: float) -> float:
        # simple nearest-rank percentile
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(lengths_sorted[k])

    return {
        "n": float(n),
        "min": float(lengths_sorted[0]),
        "mean": float(sum(lengths_sorted) / n),
        "median": float(statistics.median(lengths_sorted)),
        "p95": percentile(0.95),
        "max": float(lengths_sorted[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to JSONL file (one object per line)")
    ap.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="HF tokenizer model id to use for accurate token counts",
    )
    ap.add_argument(
        "--text_mode",
        default="plain",
        choices=["plain"],
        help="How to convert messages to text for counting",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0, only process first N rows (quick test)",
    )
    ap.add_argument(
        "--also_tiktoken",
        action="store_true",
        help="Also compute tiktoken counts (rough baseline)",
    )
    ap.add_argument(
        "--tiktoken_encoding",
        default="o200k_base",
        help="tiktoken encoding name (only used if --also_tiktoken)",
    )
    args = ap.parse_args()

    path = Path(args.jsonl)
    data = read_jsonl(path)
    if args.max_rows and args.max_rows > 0:
        data = data[: args.max_rows]

    # Load tokenizers
    hf_tok = load_hf_tokenizer(args.model_id)

    tt_enc = None
    if args.also_tiktoken:
        tt_enc = load_tiktoken(args.tiktoken_encoding)

    # Count
    hf_counts: List[int] = []
    tt_counts: List[int] = []
    id_and_count: List[Tuple[str, int]] = []

    for obj in data:
        sample_id = str(obj.get("id", ""))
        messages = obj.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Sample id={sample_id} missing 'messages' list")

        text = messages_to_text(messages, mode=args.text_mode)

        c_hf = count_tokens_hf(hf_tok, text)
        hf_counts.append(c_hf)
        id_and_count.append((sample_id, c_hf))

        if tt_enc is not None:
            tt_counts.append(count_tokens_tiktoken(tt_enc, text))

    # Summaries
    hf_sum = summarize(hf_counts)
    print("\n=== HF Tokenizer Summary (recommended / model-aligned) ===")
    for k, v in hf_sum.items():
        if k == "n":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.2f}")

    if tt_enc is not None:
        tt_sum = summarize(tt_counts)
        print("\n=== tiktoken Summary (rough baseline) ===")
        for k, v in tt_sum.items():
            if k == "n":
                print(f"{k}: {int(v)}")
            else:
                print(f"{k}: {v:.2f}")

    # Show largest prompts
    id_and_count.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Top 5 longest samples (by HF tokenizer) ===")
    for sid, c in id_and_count[:5]:
        print(f"id={sid} tokens={c}")


if __name__ == "__main__":
    main()