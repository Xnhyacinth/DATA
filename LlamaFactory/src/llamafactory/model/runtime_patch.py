from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from ..extras import logging


logger = logging.get_logger(__name__)

_APPLIED_RUNTIME_PATCHES: set[str] = set()

_PATCH_SPECS: dict[str, dict[str, object]] = {
    "llama": {
        "target_module": "transformers.models.llama.modeling_llama",
        "patch_file": "modeling_llama.py",
        "exports": ["LlamaForCausalLM"],
    },
    "qwen2": {
        "target_module": "transformers.models.qwen2.modeling_qwen2",
        "patch_file": "modeling_qwen2.py",
        "exports": ["Qwen2ForCausalLM"],
    },
    "qwen2_5_vl": {
        "target_module": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "patch_file": "modeling_qwen2_5_vl.py",
        "exports": ["Qwen2_5_VLForConditionalGeneration"],
    },
    "qwen3_vl": {
        "target_module": "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "patch_file": "modeling_qwen3_vl.py",
        "exports": ["Qwen3VLForConditionalGeneration"],
    },
    "t5": {
        "target_module": "transformers.models.t5.modeling_t5",
        "patch_file": "modeling_t5.py",
        "exports": ["T5ForConditionalGeneration"],
    },
}


def _normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(".", "_").replace("/", "_")


def _get_patch_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "patch"


def _load_patch_module(patch_key: str):
    patch_spec = _PATCH_SPECS[patch_key]
    patch_file = _get_patch_dir() / str(patch_spec["patch_file"])
    if not patch_file.exists():
        raise FileNotFoundError(f"Runtime patch file not found: {patch_file}")

    target_module = str(patch_spec["target_module"])
    fake_module_name = f"{target_module}__data_runtime_patch"
    if fake_module_name in sys.modules:
        return sys.modules[fake_module_name]

    module_spec = importlib.util.spec_from_file_location(fake_module_name, patch_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot create runtime patch module spec for: {patch_file}")

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[fake_module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _apply_runtime_patch(patch_key: str) -> None:
    if patch_key in _APPLIED_RUNTIME_PATCHES:
        return

    patch_spec = _PATCH_SPECS[patch_key]
    target_module = importlib.import_module(str(patch_spec["target_module"]))
    patch_module = _load_patch_module(patch_key)
    transformers = importlib.import_module("transformers")

    for export_name in patch_spec["exports"]:
        if hasattr(patch_module, export_name):
            patched_symbol = getattr(patch_module, export_name)
            setattr(target_module, export_name, patched_symbol)
            setattr(transformers, export_name, patched_symbol)

    _APPLIED_RUNTIME_PATCHES.add(patch_key)
    logger.info_rank0(f"Applied in-memory transformers patch: {patch_key}")


def apply_data_runtime_patches(model_type: str, model_name: str) -> None:
    candidates = [_normalize_model_name(model_type), _normalize_model_name(model_name)]
    patch_keys: list[str] = []

    if any("qwen3_vl" in candidate for candidate in candidates):
        patch_keys.append("qwen3_vl")
    elif any("qwen2_5_vl" in candidate for candidate in candidates):
        patch_keys.append("qwen2_5_vl")
    elif any("t5" in candidate for candidate in candidates):
        patch_keys.append("t5")
    elif any("llama" in candidate for candidate in candidates):
        patch_keys.append("llama")
    elif any("qwen2" in candidate for candidate in candidates):
        patch_keys.append("qwen2")

    for patch_key in patch_keys:
        _apply_runtime_patch(patch_key)
