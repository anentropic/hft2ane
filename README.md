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
- **BERT** (TODO: cross-attention, i.e. EncoderDecoderModel support)
- **RoBERTa** (TODO: CausalLM and EncoderDecoderModel support)

## TODO

- `ane_transformers` is currently pinned to PyTorch `<=1.11.0`. This means we can't load and convert any models which use PyTorch 2+ features. See https://github.com/apple/ml-ane-transformers/pull/3
  - due to bugs in their DistilBERT, and factoring out some common stuff after implementing BERT, there is very little we're importing from that lib (just the `LayerNormANE` class and the `compute_psnr` test util)... we could easily just vendor those in and drop the dependency
  - there's a few places where PyTorch 2's new `squeeze` with tuple of dims would allow us to remove a double squeeze
- Can we make use of this https://github.com/huggingface/exporters ?

### NOTE re asitop logs

These can accumulate massively... I just deleted > 40GB of asitop logs from `/private/tmp` (!)

If you find yourself running out of storage:
- `brew install ncdu`
- `sudo ncdu /private`
- `d` to delete
