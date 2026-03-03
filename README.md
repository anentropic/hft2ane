# 🤗🤖➔🍏🧠 &nbsp; `hft2ane`
## HuggingFace Transformers ➔ to ➔ Apple Neural Engine

This tool allows you to convert pre-trained models (having Transformer achitecture) from Hugging Face Hub into a form that will run on the Neural Engine of Apple Silicon Macs (and iPhone too, but have not tested).

### How does it work?

Currently the only way to run an ML model on the Neural Engine (aka ANE, aka NPU), found in Apple Silicon devices such as M1 Macs, is to convert it to CoreML format and execute it with the CoreML compiler.

The CoreML compiler will analyse the code and decide whether it is suitable for running on the ANE, otherwise it will fall-back to CPU.  A couple of pre-requisites for ANE execution are float16 precision and a specific tensor shape.

Apple published [a document here](https://machinelearning.apple.com/research/neural-engine-transformers) about how to adapt Transformer models to run on the ANE. They also published [a Python library](https://github.com/apple/ml-ane-transformers) implementing this for HuggingFace `transformers` DistilBERT models, and [an exported CoreML artefact](https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english) for the `distilbert-base-uncased-finetuned-sst-2-english` Sequence Classification model.

### About this tool

There are two parts:

- Re-implementations of [various parent models](#supported-model-types), using the code from Apple `ane_transformers` library (they provide the initial conversion for `distilbert` only).
- Tool to simplify loading pre-trained weights from HuggingFace into the appropriate re-implemented model, then exporting it to Apple's CoreML `.mlpackage` format. This can be run from Python via `coremltools` or incorporated into an XCode project for Swift, iOS etc, and CoreML will run it on the Neural Engine.

### Alternatives

- https://mlc.ai/mlc-llm/ compiles any LLM from HF, for any platform (e.g. Metal backend on Mac/iPhone). Outputs an example cli chat app. Relies on TVM for compilation so no ANE backend, though GPU can be faster.
- https://github.com/huggingface/exporters (no `pip` install yet) and https://huggingface.co/spaces/huggingface-projects/transformers-to-coreml Gradio app. Nice front-end for converting HF models to CoreML. Allows setting the 'compute unit' flag, but presumably large Transformer models will not execute on the ANE since HF Hub models don't have the necessary tweaks per the Apple doc above.

## Get started

This should probably be installed via `pipx`. (But it's not yet published to PyPI...)

### Supported model types

The process of translating models from HF `transformers` into ANE-friendly form is manual and a bit tedious. Pull requests implementing further model types are very welcome!

Currently `hft2ane` supports:

- **DistilBERT**
- **BERT** (TODO: EncoderDecoderModel support, needs translatedcross-attention implementation)
- **RoBERTa** (TODO: CausalLM and EncoderDecoderModel support)

TODO: export of `*ForMultipleChoice` models is currently failing in all cases, an issue at JIT tracing step (turns out `model.dummy_inputs` is insufficient for this use case).

EncoderDecoderModels are also a problem for HF `exporters` project - they [resort](https://github.com/huggingface/exporters/blob/main/src/exporters/coreml/__main__.py#L54) to exporting separate 'encoder' and 'decoder' .mlpackage files, which I guess you can then glue together yourself with a [CoreML pipeline](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.pipeline).

#### NOTE

Only PyTorch models are supported.

### Results

Apple report "up to" 10x speedup for their ANE-optimised DistilBERT, in the form of these hard-to-read graphs https://machinelearning.apple.com/research/neural-engine-transformers#figure2 - we can see that actual speedup is somewhat dependent on batch size and sequence length (more speedup with longer sequences).

We do a basic sanity check after export from `hft2ane` using a batch of 1 and very short sequence. Our version of DistilBERT is the same as Apple's one above and reports a **3.67x** speedup for this check on my 2020 M1 Macbook Air. This is comparing the compiled CoreML models with or without the ANE compute unit flag enabled. We can use that as a baseline.

I converted `dslim/bert-base-NER-uncased` as `BertForTokenClassification.mlpackage` and measured a **3.26x** speedup.

I converted `deepset/roberta-base-squad2` as `RobertaForQuestionAnswering.mkpackage` and measured **3.53x** speedup.

So... these translations look basically successful!

## CLI usage

The CLI has two subcommands: `convert` and `verify`.

### Convert

Convert a HuggingFace model to an ANE-optimised CoreML `.mlpackage`:

```bash
# Interactive — prompts guide you through model selection
poetry run python -m hft2ane.cli convert

# Specify a model directly
poetry run python -m hft2ane.cli convert bert-base-uncased

# With options
poetry run python -m hft2ane.cli convert bert-base-uncased \
  --seq-len 256 \
  --out-dir ./converted \
  --pkg-name bert.mlpackage

# Use a specific model class
poetry run python -m hft2ane.cli convert bert-base-uncased \
  --model-cls BertForSequenceClassification
```

After conversion, a verification step runs automatically (see below).

### Verify

Verify an already-converted model matches the original HF model's outputs:

```bash
poetry run python -m hft2ane.cli verify ./converted/bert.mlpackage

# Also confirm it actually runs on the Neural Engine (requires sudo)
sudo poetry run python -m hft2ane.cli verify ./converted/bert.mlpackage --confirm-ane
```

Verification does the following:
- **Sanity check** — compares converted model outputs against the original, checking they agree within tolerance.
- **Speedup measurement** — runs inference 100 times with ANE enabled vs disabled and reports the speedup ratio. Always runs.
- **ANE confirmation** (optional, `--confirm-ane`, requires sudo) — uses Apple `powermetrics` to check that the Neural Engine is actually being used during inference.

## TODO

- The sequence length gets baked into the exported model. HF exporters provides for variable sequence lengths, but we run into this issue https://github.com/apple/coremltools/issues/1763
  - needs to be exposed as a cli param

### NOTE re asitop logs

These can accumulate massively... I just deleted > 40GB of asitop logs from `/private/tmp` (!)

If you find yourself running out of storage:
- `brew install ncdu`
- `sudo ncdu /private`
- `d` to delete
