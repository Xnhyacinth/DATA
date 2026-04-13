"""Lightweight helpers for CofeAI/FLM-Audio integration.

Keep this module dependency-free (no torch/transformers imports) so it can be
used safely across hparams/data/model layers without creating heavy import
chains or cycles.
"""
 
from __future__ import annotations
 
from typing import Any
 
 
def normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(".", "_").replace("/", "_")
 
 
def is_flm_audio_model(model_name_or_path: str) -> bool:
    normalized = normalize_model_name(model_name_or_path)
    return ("cofeai_flm_audio" in normalized) or normalized.endswith("flm_audio")
 
 
def has_audio_feature_extractor(processor: Any) -> bool:
    """Return True if processor looks like it has an audio feature extractor."""
    if processor is None:
        return False
    return (getattr(processor, "feature_extractor", None) is not None) or (
        getattr(processor, "audio_processor", None) is not None
    )


def get_primary_architecture(config: Any, fallback_name: str = "") -> str:
    """Best-effort architecture name extraction for remote-code and wrapped models."""
    if config is not None:
        architectures = getattr(config, "architectures", None) or []
        if architectures:
            return str(architectures[0]).lower()

        model_type = getattr(config, "model_type", None)
        if model_type:
            return normalize_model_name(str(model_type))

    return normalize_model_name(fallback_name)


def is_t5_like_architecture(config: Any, fallback_name: str = "") -> bool:
    return "t5" in get_primary_architecture(config, fallback_name=fallback_name)
