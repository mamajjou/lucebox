# Laguna mixed staged prefill

This is an opt-in single-process mixed engine for VRAM-constrained Laguna MoE inference.

It does **not** copy KV state between buun and Luce, load the model twice, or re-run RoPE.
It keeps Luce's model, attention implementation, KV cache, and hybrid decode engine in one
backend.

## Execution model

```text
prompt
  -> Luce attention/router, layer by layer
  -> temporarily stage one complete 256-expert layer on GPU
  -> full-stack batched GPU mul_mat_id prefill
  -> restore that layer's calibrated hot experts + empty cache slots
  -> next layer
  -> existing Luce hot-GPU/cold-CPU hybrid decode
```

Only one full expert layer is staged at a time. The layer's normal hot buffer is released
first, so the temporary VRAM increase is approximately the cold fraction of one layer plus
compute workspace, rather than another copy of the model.

## Enable

```bash
export DFLASH_LAGUNA_MIXED_PREFILL=1
export DFLASH_LAGUNA_AUTO_HEAD_MAJOR=0   # required on HIP/gfx1100

# Optional controls
export DFLASH_LAGUNA_MIXED_PREFILL_CHUNK=256
export DFLASH_LAGUNA_MIXED_RESERVE_MB=384
export DFLASH_LAGUNA_MIXED_VERBOSE=1
```

`DFLASH_LAGUNA_MIXED_PREFILL_FALLBACK=1` explicitly permits falling back to the old hybrid
prefill path after a recoverable staging failure. It is off by default so a failed mixed run
does not silently turn into another multi-minute prompt ingestion.

Without `DFLASH_LAGUNA_MIXED_PREFILL=1`, `LagunaMixedBackend` delegates directly to the
existing `LagunaBackend` and changes no runtime behavior.

## Required RX 7900 XTX validation

Use a clean HIP/gfx1100 build based on `c931461` or later.

```bash
rm -rf build-hip-gfx1100
cmake -S server -B build-hip-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip-gfx1100 --target bench_laguna_spark dflash_server -j16
```

Start with plain hybrid placement and no expert cache:

```bash
DFLASH_LAGUNA_MIXED_PREFILL=1 \
DFLASH_LAGUNA_AUTO_HEAD_MAJOR=0 \
DFLASH_EXPERT_BUDGET_PCT=60 \
build-hip-gfx1100/bench_laguna_spark \
  /path/to/Laguna-S-2.1-UD-IQ1_S.gguf \
  128 64 --max-ctx 2048 --kv q8_0
```

Then validate prompt sizes `1`, `128`, `2048`, and the exact 8.8K Pi fixture. Record:

- prefill tokens/s;
- decode tokens/s;
- peak VRAM;
- staged upload time per layer;
- output equality against ordinary hybrid for a deterministic short fixture;
- whether the restored decode placement reports the original hot/cold counts.

After plain placement passes, repeat with the recorded hotness profile and Spark cache.

## Safety and fallback properties

- Full-stack staging reads from the retained GGUF mmap already owned by hybrid storage.
- Every layer restores the persistent pinned-hot allocation before the next layer and before
  decode.
- Cache-ring slots are restored empty and zero-initialized; stale swapped-in experts are not
  trusted after staging.
- A VRAM guard checks the complete layer size plus a configurable compute reserve after the
  persistent hot buffer is released.
- Restore failure is fatal and disables automatic fallback because decode placement can no
  longer be assumed valid.
- Compact SWA rings are not active in Laguna hybrid mode. HIP must use the legacy per-head KV
  layout; the head-major path remains unsafe on gfx1100.

## Expected result

The goal is not necessarily to equal buun's best 900–1000 prompt tokens/s immediately. The
first acceptance gate is that an 8–10K prompt completes in practical time while decode stays
near the already measured 40–55 tokens/s. The staged design removes the known reduced-stack
`<=4`-token MoE sub-batching from prefill and avoids the incompatible cross-backend KV handoff.
