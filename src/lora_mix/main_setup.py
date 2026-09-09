# main_setup.py
import os
import torch
import json
from peft import PeftMixedModel

from lora_mix.base import get_mix_handler
from lora_mix import average_merge, cat_layer, loraflow, loraall, comol, asymcore  # noqa: F401


def _setup_cat_lora_tuning(
    config,
    model,
    model_args,
    finetuning_args,
    training_args,
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
):
    if model_args.adapter_name_or_path is None:
        return model
    adapters_to_load = model_args.adapter_name_or_path
    if isinstance(adapters_to_load, str):
        adapters_to_load = [x.strip() for x in adapters_to_load.split(",") if x.strip()]
        model_args.adapter_name_or_path = adapters_to_load

    if not isinstance(adapters_to_load, (list, tuple)) or len(adapters_to_load) < 2:
        raise ValueError(f"[LoRA-Mix] adapter_name_or_path must contain at least 2 LoRA paths, got: {adapters_to_load}")

    init_kwargs = {
        "subfolder": model_args.adapter_folder,
        "offload_folder": model_args.offload_folder,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    model = PeftMixedModel.from_pretrained(
        model,
        adapters_to_load[0],
        adapter_name="adapter_0",
        is_trainable=False,        **init_kwargs,
    )

    for i, adapter in enumerate(adapters_to_load[1:], start=1):
        model.load_adapter(adapter, adapter_name=f"adapter_{i}", is_trainable=False, **init_kwargs)

    adapter_names = [f"adapter_{i}" for i in range(len(adapters_to_load))]
    model.set_adapter(adapter_names)

    mix_mode = getattr(finetuning_args, "lora_mix_mode", "default").lower()
    print(f"[LoRA-Mix] setup mix_mode = {mix_mode}")
    # Never silently substitute another algorithm.
    # An unknown mode must fail explicitly to preserve experiment validity.
    handler = get_mix_handler(mix_mode)

    model = handler(model, adapters_to_load, is_trainable, finetuning_args, training_args)


    # Load trainable routing state during inference.
    if (
        not getattr(training_args, "do_train", False)
        and mix_mode in {"cat_layer", "loraflow", "loraall"}
    ):
        state_path = getattr(
            finetuning_args,
            "lora_mix_state_path",
            None,
        )
        if not state_path:
            raise RuntimeError(
                f"[{mix_mode}] Prediction requires "
                "lora_mix_state_path."
            )

        from lora_mix_save.generic_mix_state import (
            load_generic_mix_state,
        )
        load_generic_mix_state(
            model=model,
            mix_mode=mix_mode,
            state_path=state_path,
            output_dir=training_args.output_dir,
        )

    model.lora_mix_mode = mix_mode

    if not hasattr(model, "get_base_model"):
        model.get_base_model = lambda: model.base_model

    mode_map = {
        "default": 0,
        "average_merge": 16,
        "cat_layer": 5,
        "loraflow": 6,
        "loraall": 10,
        "comol": 12,
        "asymcore": 14,
    }
    mode_id = torch.tensor(mode_map.get(mix_mode, 0), dtype=torch.int32)
    model.register_buffer("lora_mix_mode_buf", mode_id)
    print(f"[LoRA-Mix] Global mode buffer registered: {mix_mode} ({int(mode_id)})")

    for module in model.modules():
        module._get_mix_mode = lambda mm=mode_id: mm
    print(f"[LoRA-Mix] Bound _get_mix_mode() for all {len(list(model.named_modules()))} modules.")

    return model
