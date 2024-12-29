# HuggingFaceTransformers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/HuggingFaceTransformers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/HuggingFaceTransformers.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/HuggingFaceTransformers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/HuggingFaceTransformers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/HuggingFaceTransformers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/HuggingFaceTransformers.jl)


A light wrapper around HuggingFace's Transformers Python library, with very little functionality directly supported. Can load autoregressive LLMs, tokenizers, and generate text.

## Installation:

```julia
] add https://github.com/MurrellGroup/HuggingFaceTransformers.jl
```

## Usage:

```julia
using HuggingFaceTransformers

tokenizer = from_pretrained(AutoTokenizer, "HuggingFaceTB/SmolLM2-135M");
model = from_pretrained(CausalLM, "HuggingFaceTB/SmolLM2-135M");

generate(
    model,
    "Python is the worst language for deep learning because",
    tokenizer,
    max_length=15,
    pad_token_id=encode(tokenizer, "<|endoftext|>")[1]
)
```

Or if you want to handle the tokens explicitly:

```julia
text = "Python is the worst language for deep learning because"

toks = encode(tokenizer, text)

gen = generate(
    model,
    toks,
    max_length=15,
    pad_token_id=encode(tokenizer, "<|endoftext|>")[1]
)

println(decode(tokenizer, gen))
```

Using "chat templates" for instruct models:

```julia
model = from_pretrained(CausalLM, "HuggingFaceTB/SmolLM2-1.7B-Instruct");
tokenizer = from_pretrained(AutoTokenizer, "HuggingFaceTB/SmolLM2-1.7B-Instruct")
prompt = "May a moody baby doom a yam?"

messages = [
    Dict("role" => "system", "content" => "You are Bob, a helpful assistant."),
    Dict("role" => "user", "content" => prompt)
]

toks = apply_chat_template(
    tokenizer,
    messages,
    add_generation_prompt=true
)

gen = generate(
    model,
    toks,
    max_length=500,
    pad_token_id=encode(tokenizer, "<|endoftext|>")[1]
);
println(decode(tokenizer, gen, skip_special_tokens=true));
```

## Other models

This also works with other kinds of models, but you sometimes have to get a bit closer to the Python.

This is a HuggingFace port of [ESM-C](https://huggingface.co/Synthyra/ESMplusplus_large), a protein language model:

```julia
model = from_pretrained(Model, "Synthyra/ESMplusplus_large", trust_remote_code=true)
#Note the tokenizer loads differently - directly from the model.
tokenizer = AutoTokenizer(model.py_transformer.tokenizer)

#We can run this using the provided encode wrapper for a signle sequnece (where toks is a Julia Array):
toks = encode(tokenizer, "MPRTEIN");
o = model(toks)
#tensor will detach the PyTorch tensor, and convert it to a Julia array.
tensor(o.last_hidden_state)

#...but if you want to batch, you need to use the Python tokenizer directly:
pytoks = tokenizer.py_tokenizer(("MPRTEIN","MSEQWENCE"), padding=true, return_tensors="pt");
o = model(pytoks.input_ids);
tensor(o.last_hidden_state)
```