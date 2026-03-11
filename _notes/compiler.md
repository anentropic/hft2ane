## Problem

Apple `coremltools` Python lib is currently the only way to convert PyTorch models to `.mlpackage` compiled for ANE (?)

But broadly speaking the momentum of Apple AI development is behind the MLX ecosystem now.

`coremltools` is behind on Python and PyTorch versions. And there are unresolved bugs in converting 'stateful' models (i.e. anything with a KV cache, i.e. any chat model).

So currently we are limited in types of model we can convert and compile to ANE, basically BERTs, embedding models, maybe some vision models.

## Custom compiler?

https://maderix.substack.com/p/inside-the-m4-apple-neural-engine?r=1afltp
https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615?r=1afltp
https://github.com/maderix/ANE

In light of the reverse-engineering work above...

Is there enough info to make a custom compiler and bypass `coremltools`?

I would guess we need to first go from (hft2ane-optimised...) PyTorch to some intermediate format (ONNX? Triton? what else?) and then the new compiler would take that and produced something that runs on the ANE.

### Triton

At a glance looks potentially good for this.

- https://pytorch.org/blog/triton-kernel-compilation-stages/
- https://triton-lang.org/main/dialects/dialects.html
  - ...we could make some kind of ANE dialect?
