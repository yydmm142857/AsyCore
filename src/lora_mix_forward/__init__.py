# lora_mix_forward/__init__.py
import importlib
import pkgutil

__all__ = {}

def _auto_import_forward_modules():
    package = __name__
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name.startswith("forward_") or module_name.endswith("_forward"):
            full_name = f"{package}.{module_name}"
            module = importlib.import_module(full_name)
            for attr_name in dir(module):
                if attr_name.startswith("forward_"):
                    __all__[attr_name.replace("forward_", "")] = getattr(module, attr_name)

_auto_import_forward_modules()
