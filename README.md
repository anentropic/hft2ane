# 🤗🤖➔🍏🧠
## `hft2ane`: HuggingFace Transformers to Apple Neural Engine

This tool allows you to convert pre-trained models (having Transformer achitecture) from Hugging Face Hub into a form that will run on the Neural Engine of Apple Silicon Macs (and iPhone too, but have not tested).

### How does it work?

Currently the only way to run an ML model on the Neural Engine (aka ANE, aka NPU), found in Apple Silicon devices such as M1 Macs, is to convert it to CoreML format and execute it with the CoreML compiler.

The CoreML compiler will analyse the code and decide whether it is suitable for running on the ANE, otherwise it will fall-back to CPU.  A couple of pre-requisites for ANE execution are float16 precision and a specific tensor shape.

Apple published [a document here](https://machinelearning.apple.com/research/neural-engine-transformers) about how to adapt Transformer models to run on the ANE. They also published [a Python library](https://github.com/apple/ml-ane-transformers) implementing this for HuggingFace `transformers` DistilBERT models, and [an exported CoreML artefact](https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english) for the `distilbert-base-uncased-finetuned-sst-2-english` Sequence Classification model.

### About this tool

There are two parts:

- Re-implementations of [various parent models](#supported-model-types), using the code from Apple `ane_transformers` library (they provide the initial conversion for `distilbert` only).
- Tool to simplify loading pre-trained weights from HuggingFace into the appropriate re-implemented model, then exporting it to Apple's CoreML `.mlpackage` format. This can be run from Python via `coremltools` or incorporated into an XCode project for Swift, iOS etc, and CoreML will run it on the Neural Engine.


## Get started

This should probably be installed via `pipx`.

### Supported model types

The process of translating models from HF `transformers` into ANE-friendly form is manual. Pull requests implementing further model types are very welcome!

Currently `hft2ane` supports:

- DistilBERT

## TODO

- `ane_transformers` is currently pinned to PyTorch `<=1.11.0`. This means we can't load and convert any models which use PyTorch 2+ features. See https://github.com/apple/ml-ane-transformers/pull/3
  - it also has another bug we already monkey-patched around, the future might be to vendor a forked version into the project
- Can we make use of this https://github.com/huggingface/exporters