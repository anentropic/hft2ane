# Maintainer notes

## Project structure

The project converts HuggingFace Transformer models to ANE-optimised CoreML format. The main components:

- `hft2ane/cli.py` — CLI entry point with `convert` and `verify` subcommands. Uses `beaupy` for interactive TUI when no model name is provided.
- `hft2ane/convert/convert.py` — Core conversion logic. Two code paths:
  - `hf_to_coreml` — newer path using the vendored HF `exporters` library directly.
  - `to_coreml_internal` — older path that predates the exporters integration.
- `hft2ane/evaluate/evaluate.py` — Post-conversion evaluation utilities (not exposed as a standalone CLI command, only used internally by `_verify` in `cli.py`).
- `hft2ane/mappings.py` — Registry of supported models and their task mappings.

### Vendored dependencies

- `vendor/exporters/` — Fork of [huggingface/exporters](https://github.com/huggingface/exporters) with project-specific modifications:
  - `CoreMLConfig` has an `overrides`/`sequenceLength` mechanism for baking sequence length into exported models.
  - `FeaturesManager` uses `@classmethod` methods and includes RoBERTa feature mappings.
  - `BertCoreMLConfig` has a custom `atol_for_validation`.
- `vendor/ml-ane-transformers/` — Apple's [ml-ane-transformers](https://github.com/apple/ml-ane-transformers) library providing the ANE-optimised building blocks.

### Conversion pipeline

`hf_to_coreml` flow:
1. Resolve the `CoreMLConfig` for the model/task via `FeaturesManager`.
2. Load a preprocessor (tokenizer/feature extractor).
3. Call the vendored `export()` which traces the model and converts to CoreML.
4. `_set_metadata` stamps HF Hub metadata (author, license, model name, sequence length) onto the `.mlpackage`.

`_set_metadata` accepts `sequence_length: int | None = None` — the `hf_to_coreml` path passes it from the config, while the older `to_coreml_internal` path omits it.

## Evaluation internals

`hft2ane/evaluate/evaluate.py` is only consumed by `cli.py`'s `_verify` function. It provides:

### `sanity_check` (currently commented out in `_verify`)
Compares converted CoreML model outputs against the original HF model using dummy inputs. Checks both relative tolerance (`rtol=0.1`) and peak signal-to-noise ratio (60dB threshold, via Apple's `compute_psnr`). The active code path now uses `validate_model_outputs` from the vendored exporters instead.

### `measure_ane_speedup_from_converted`
Benchmarks inference (100 iterations) with `ComputeUnit.ALL` (ANE enabled) vs `ComputeUnit.CPU_AND_GPU` (ANE disabled). Returns the speedup ratio — values above ~1.5x indicate the model is benefiting from ANE execution. Always runs during verification.

### `confirm_ane_via_powermetrics`
Uses Apple's `powermetrics` (requires sudo) via `asitop` to measure ANE power draw. Spawns a subprocess that collects power readings, takes a baseline, then runs inference and checks if ANE wattage rises above zero. Only runs when `--confirm-ane` is passed.

**Caution:** `asitop` logs can accumulate massively in `/private/tmp` (40GB+). See README for cleanup instructions.

## Testing

```bash
# Collect tests (structural check — imports, no broken references)
uv run pytest tests/ --collect-only

# Run tests (requires macOS with coremltools for test_bert.py)
uv run pytest tests/

# Pre-commit hooks
prek run --all-files
```

Note: `test_bert.py` imports `coremltools` at module level so it will fail to collect on non-macOS or without `coremltools` installed. The other test files (distilbert, roberta) collect and run without it.
