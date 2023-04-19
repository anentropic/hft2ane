# torchinfo recipes

We have to pass some dummy data in, presumably so it can trace through a forward pass, forcing PyTorch to materialise the model.

### `t5-large` recipe:

```python
import torch
import torchinfo
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

config = T5Config.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# this will be different for different model types:
input_data = (torch.ones(1, config.max_length, dtype=torch.int),) * 3

summary = torchinfo.summary(model, input_data=input_data, device="cpu")
```

(we learn the required shape of `input_data` from this comment: https://stackoverflow.com/questions/65140400/valueerror-you-have-to-specify-either-decoder-input-ids-or-decoder-inputs-embed#comment115193484_65140400)

You can `repr(summary)` to get a nice table, but it is also an object with methods.

```
Total params: 1,138,405,888
Params size (MB): 3082.27
Estimated Total Size (MB): 3194.34
```
...suggests that this is a 1.1B model (not 770M as I had read elsewhere).

This might be useful in some way https://github.com/Ki6an/fastT5 in future.

### `gpt2-xl` recipe

```python
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

summary = torchinfo.summary(
    model,
    input_data=torch.ones(1, config.max_length, dtype=torch.int),
    device="cpu",
)
```

showing:
```
Total params: 1,638,022,400
Params size (MB): 6552.09
Estimated Total Size (MB): 6696.07
```

Presumably these are float32 weights currently, hence ~4x no. of params.

Anyway, looks like T5-large may be possible, since it's smaller than this.

I'm not sure how this fits with https://github.com/smpanaro/more-ane-transformers/blob/main/src/experiments/NOTES.md where they get `gpt2-xl` running on ANE, while also identifying a 3GB size limit for models to run on it.  I don't quite follow what is the pipeline trick they talk about doing (seems to involve breaking the model into chunks?). It may be a 3Gi limit (3Gi is approximately 3.221 GB).


NOTE: [CoreML Pipeline](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.pipeline) seems different to [HuggingFace Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)

In CoreML the Pipeline is a sequence of models. In HuggingFace it seems a high-level abstraction for common tasks against a single model, simplifying preparing and passing the input and getting the desired output.

FWIW I think there's room for building the HF style abstraction on top of CoreML for running these models
