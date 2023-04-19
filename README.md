# Apple Neural Engine Tranformers Zoo (ANETZ)
This tool provides for converting pre-trained models from Hugging Face Hub into a form that will run on the Neural Engine of Apple Silicon Macs (and iPhone too, but not tested).

There are two parts:
- Re-implementations of various parent models, using the code from Apple `ane_transformers` library (they provide the initial conversion for `distilbert` only)
- Tool to simplify loading pre-trained weights from HuggingFace into the appropriate re-implemented model, then exporting it to Apple's CoreML `.mlpackage` format. This can be run from Python via `coremltools` or incorporated into an XCode project for Swift, iOS etc, and CoreML will run it on the Neural Engine.

It very much builds on the work in https://github.com/apple/ml-ane-transformers
