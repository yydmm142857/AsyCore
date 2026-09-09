# Methods and configuration

## AsyCore

The main method uses module-wise routing from the current LoRA-attached module
input and shares the right core factor across experts:

```yaml
lora_mix_mode: asymcore
asymcore_route_granularity: module
asymcore_factor_variant: right_shared
asymcore_gate_mode: raw
asymcore_train_shared: true
```

The structural and routing variants use the same implementation. Unlisted
AsyCore fields remain equal to the main configuration above.

| Method | Structural setting | Function |
|---|---|---|
| AsyCore | `module`, `right_shared`, `raw`, shared side trainable | Main asymmetric-core method |
| AsyCore-F | `asymcore_train_shared: false` | Freezes the shared right core after least-squares initialization |
| AsyCore-Left | `asymcore_factor_variant: left_shared` | Shares the left core and retains expert-specific right cores |
| AsyCore-Both | `asymcore_factor_variant: both_diff` | Retains expert-specific core factors on both sides |
| AsyCore-V | `asymcore_gate_mode: vproj` | Routes from the input projected onto the right reference basis |
| AsyCore-R | `asymcore_gate_mode: rproj` | Further projects routing features through the shared right core |
| AsyCore-Layer | `asymcore_route_granularity: layer` | Uses one token-dependent routing result for all LoRA modules in a layer |
| AsyCore-Layer-LR | layer routing and `asymcore_layer_gate_residual_rank: 15` | Adds a rank-15 residual branch to match the module-routing parameter scale |

Layer-wise routing uses raw layer inputs. The `vproj` and `rproj` routing modes
are defined for module-wise routing because their reference-space projections
depend on the current LoRA module.

## Fusion baselines

| Method in the paper | `lora_mix_mode` | Implementation |
|---|---|---|
| Average Merge | `average_merge` | Equal-weight LoRA update merging |
| CAT | `cat_layer` | One learned expert-weight vector per Transformer layer |
| LoRA-Flow | `loraflow` | Token-wise routing shared across modules in a layer |
| LoRA-All | `loraall` | Module-wise token-dependent routing |
| CoMoL | `comol` | Dynamic fusion through trainable core-space reconstruction |

The CoMoL controls reported in the structural comparison use the same handler:

| Method | Configuration change | Function |
|---|---|---|
| CoMoL-F | `comol_freeze_shared_ab: true` | Freezes the shared high-dimensional reconstruction bases |
| CoMoL-Raw | `comol_gate_mode: raw` | Routes directly from the original module input instead of the projected core representation |

## Structural option reference

| Option | Values used in the paper | Structural role |
|---|---|---|
| `lora_mix_mode` | `average_merge`, `cat_layer`, `loraflow`, `loraall`, `comol`, `asymcore` | Selects the fusion implementation |
| `asymcore_route_granularity` | `module`, `layer` | Chooses whether routing weights are generated independently per LoRA module or shared within a Transformer layer |
| `asymcore_factor_variant` | `right_shared`, `left_shared`, `both_diff` | Selects which low-dimensional core side is shared across experts |
| `asymcore_gate_mode` | `raw`, `vproj`, `rproj` | Selects the representation supplied to each routing gate |
| `asymcore_train_shared` | `true`, `false` | Enables or freezes optimization of the shared core factor |
| `asymcore_layer_gate_residual_rank` | `0`, `15` | Controls the optional low-rank residual branch used only for parameter-matched layer routing |
| `comol_freeze_shared_ab` | `false`, `true` | Enables or freezes CoMoL's shared high-dimensional reconstruction matrices |
| `comol_gate_mode` | `vproj`, `raw` | Routes from the projected core representation or the original module input |
| `comol_ref_adapter_idx` | expert index | Selects the expert used to initialize CoMoL's shared structure |
