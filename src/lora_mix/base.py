# lora_mix/base.py
MIX_MODE_HANDLERS = {}

def register_mix_mode(name):
    def wrapper(fn):
        MIX_MODE_HANDLERS[name] = fn
        return fn
    return wrapper

def get_mix_handler(name: str):
    if name not in MIX_MODE_HANDLERS:
        raise ValueError(f"[LoRA-Mix] Unknown mix_mode={name!r}")
    return MIX_MODE_HANDLERS[name]
