# lora_mix_save/__init__.py
import importlib
import pkgutil
import logging

__all__ = {}

def _auto_import_save_modules():
    package = __name__
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name.endswith("_weight") or module_name.endswith("_weights"):
            full_name = f"{package}.{module_name}"
            module = importlib.import_module(full_name)
            for attr_name in dir(module):
                if attr_name.startswith("save_"):
                    mode_name = attr_name.replace("save_", "")
                    for suffix in ["_gate_params", "_weights", "_weight", "_params"]:
                        if mode_name.endswith(suffix):
                            mode_name = mode_name[: -len(suffix)]
                            break
                    __all__[mode_name] = getattr(module, attr_name)

    logging.info(f"[LoRA-Mix] Auto-loaded save modes: {list(__all__.keys())}")

if not __all__:
    _auto_import_save_modules()
