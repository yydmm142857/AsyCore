# Integration patches

These patches capture the minimal integration changes relative to
LLaMA-Factory commit `ea31c43d806162a7fd98065abfef2d974fff5766`.

Apply them from the root of that exact checkout:

```bash
for patch_file in /path/to/AsyCore/patches/*.patch; do
  patch -p1 < "$patch_file"
done
```

The patches add fusion configuration arguments, multi-adapter setup,
trainable-state loading and saving, optimizer parameter groups, and the
gradient-checkpointing compatibility path used by AsyCore.
