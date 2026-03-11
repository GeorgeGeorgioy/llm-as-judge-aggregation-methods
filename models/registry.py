# models/registry.py

"""
_normalize: internal helper to normalize model alias by lowercasing and removing spaces, dashes, and underscores.
MODEL_ALIASES: mapping from normalized alias to actual model ID used by vLLM server.
normalize_model_alias: public function to normalize a model alias. used for looking up MODEL_RUNNERS.
resolve_model_id: public function to resolve a user-provided model alias to the actual model ID
needed to start the vLLM server. raises ValueError if alias is unknown.
"""


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


MODEL_ALIASES = {
    "qwen7": "Qwen/Qwen2.5-7B-Instruct",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama8": "meta-llama/Llama-3.1-8B-Instruct",
}


def normalize_model_alias(alias: str) -> str:
    return _normalize(alias)

def resolve_model_id(alias: str) -> str:
    key = _normalize(alias)

    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model alias '{alias}'. "
            f"Available: {list(MODEL_ALIASES.keys())}"
        )

    return MODEL_ALIASES[key], key