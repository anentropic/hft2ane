# Models to convert

Ones that look interesting to me...

## BERTs

- **BERT:** the OG
  - https://huggingface.co/mosaicml/mosaic-bert-base an improved BERT: "MosaicBERT trains faster and achieves higher pretraining and finetuning accuracy when benchmarked against [HF bert-base]
- **DistilBERT:** reduced params by training under the guidance of a BERT tutor
- **RoBERTa:** minor tweaks and an improved training regime, fine-tuned large RoBERTas are SOTA on many benchmarks.
- **ALBERT:** reduced params and model tweaks. ALBERT-xxl (235M) beats BERT-large (335M) and RoBERTa-base.
- https://huggingface.co/microsoft/graphcodebert-base
  A 'Graphformer' (GNN-nested Transformers) "which also considers data-flow information along with code sequences" ... "trained on the CodeSearchNet dataset, which includes 2.3M functions with document pairs for six programming languages". See also:
  - https://github.com/microsoft/CodeBERT
    the repo contains GraphCodeBERT plus additional models below
  - https://arxiv.org/pdf/2009.08366.pdf "GraphCodeBERT, a pre-trained model for programming language that considers the inherent structure of code". The graphs in training are based on dataflow analysis, rather than AST (which usually has an unnecessrily deep and noisy structure). In their results it outperforms CodeBERT below, as well as a RoBERTa pre-trained on code.
    - this looks like the raw dataset https://huggingface.co/datasets/code_search_net ...this doesn't have the dataflow graph though
    - the paper doesn't say exactly how the dataflow graphs are generated, but looks like the code is here: https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch/parser
    - more info here: https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch
      - it's RoBERTa with a bit of extra wrapping code in an `nn.Module` https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/model.py ...I think this could be uploaded as an HF model with custom code. Or at least any ANE conversion should include it.
    - https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/translation
    - https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement
    - https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/clonedetection
  - https://huggingface.co/microsoft/codebert-base a RoBERTa trained on code
  - https://huggingface.co/microsoft/unixcoder-base model card says its parent is RoBERTa
  - https://huggingface.co/microsoft/codereviewer a T5 transformer model for code reviews
- **UniRel:** extract knowledge-graph triples from text

### Not BERTs


- **(Flan-)T5** and derivatives
  - https://github.com/bigscience-workshop/t-zero a T5 family trained for zero-shot tasks, outperforming GPT3-175B on many. Comes in 3B and 11B variants (both too big for ANE). The 3B variant is "Same as T0 but starting from a T5-LM XL (3B parameters) pre-trained model". Training dataset is open https://huggingface.co/datasets/bigscience/P3
  - https://ai.googleblog.com/2023/02/the-flan-collection-advancing-open.html Flan-T5-XL (3B) trained on latest dataset beats instruction-tuned OPT-175B. Flan-T5-large does pretty well for some prompts I tried.
  - https://github.com/lm-sys/FastChat#FastChat-T5 -> https://huggingface.co/lmsys/fastchat-t5-3b-v1.0 a finetuned Flan-T5-XL (3B) that beats Dolly v2 on the Vicuna benchmark. too big for ANE, but maybe useful with Metal backend for local Langchain etc.
- **GPT** of course
  - see https://github.com/anentropic/experiments-coreml-gpt2#update for notes and related models
  - https://huggingface.co/MBZUAI/LaMini-GPT-1.5B somehow outperforms Alpaca LLaMA 7B on some metrics 🤯 (will be deeply underwhelming c/f ChatGTP of course). It is just GPT2-XL fine-tuned so the https://github.com/smpanaro/more-ane-transformers code should work.
  - See also:
    - https://huggingface.co/MBZUAI/LaMini-Flan-T5-783M
    - https://huggingface.co/MBZUAI/LaMini-Neo-1.3B
    - https://huggingface.co/MBZUAI/LaMini-Cerebras-1.3B
    - and smaller versions too
- **BART** is to BERT as T5 is to GPT (i.e. encoder-decoder vs encoder-only)
  - https://huggingface.co/Babelscape/rebel-large a BART that extracts knowledge-graph triples
  - https://huggingface.co/facebook/bart-large-mnli does "Zero-Shot Classification"
- **SAM (Segment Anything)**
  - "The image encoder has 632M parameters. The prompt encoder and mask decoder have 4M parameters."
  - model is Apache 2.0 license 🎉
  - https://github.com/facebookresearch/segment-anything#model-checkpoints free download
  - https://huggingface.co/docs/transformers/main/en/model_doc/sam
- **Perceiver** https://huggingface.co/docs/transformers/model_doc/perceiver
- **BLIP-2** https://huggingface.co/docs/transformers/main/model_doc/blip-2 (and BLIP) for image+text tasks
- **REALM/RAG/RETRO** retrieval-enhanced transformers... REALM and RETRO from Google (using BERT-derived retriever), RAG from Meta (using BART)
  - http://jalammar.github.io/illustrated-retrieval-transformer/
  - https://huggingface.co/docs/transformers/model_doc/realm
    - https://towhee.io/text-embedding/realm
  - https://huggingface.co/docs/transformers/main/model_doc/rag
    - https://huggingface.co/blog/ray-rag
  - RETRO is newer than REALM and RAG? The [paper](https://arxiv.org/pdf/2112.04426.pdf) shows results for 172M, 425M, 1.5B and 7.5B variants. Also performs well with retrieval disabled (e.g. out of corpus tasks). "baseline models can be rapidly fine-tuned into Retro models to obtain nearly the same performance as if trained from scratch"
    - https://github.com/lucidrains/RETRO-pytorch
    - http://mitchgordon.me/ml/2022/07/01/retro-is-blazing.html
- **Decision Transformer**  
  "Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks"
  - https://huggingface.co/docs/transformers/model_doc/decision_transformer
