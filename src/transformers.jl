abstract type HuggingFaceModel end

"""
    Tokenizer

A wrapper around a Python tokenizer.
"""
struct Model <: HuggingFaceModel
    py_transformer::Py
end

struct CausalLM <: HuggingFaceModel
    py_transformer::Py
end

struct AutoTokenizer
    py_tokenizer::Py
end

"""
    from_pretrained(Transformer, model_path::String) -> Model

Load a pretrained model. `m` can be a model name or a path to a local model, or a HuggingFace URL.
"""
from_pretrained(::Type{Model}, m::String; kwargs...) = Model(Transformers.AutoModel.from_pretrained(m; kwargs...))

"""
    from_pretrained(CausalLM, model_path::String) -> Model

Load a pretrained model. `m` can be a model name or a path to a local model, or a HuggingFace URL.
"""
from_pretrained(::Type{CausalLM}, m::String; kwargs...) = CausalLM(Transformers.AutoModelForCausalLM.from_pretrained(m; kwargs...))

"""
    from_pretrained(model_path::String) -> Tokenizer

Load a pretrained AutoTokenizer. `m` can be a model name or a path to a local model, or a HuggingFace URL.
Compared to HuggingFaceTokenizers.jl, this can use eg. chat templates.
"""
from_pretrained(::Type{AutoTokenizer}, m::String; kwargs...) = AutoTokenizer(Transformers.AutoTokenizer.from_pretrained(m; kwargs...))

"""
    apply_chat_template(tokenizer::AutoTokenizer, messages::Vector{Dict{String, String}}, tokenize=true, add_generation_prompt=false, kwargs...)

Applies a chat template (if one is associated with this model's tokenizer).
Keyword arguments include `tokenize` (bool) and `add_generation_prompt` (bool).

```julia
prompt = "Write me a poem."
messages = [
    Dict("role" => "system", "content" => "You are Bob, a helpful assistant."),
    Dict("role" => "user", "content" => prompt)]

text = apply_chat_template(tokenizer, messages, tokenize=false, add_generation_prompt=true)
```
"""
function apply_chat_template(tokenizer::AutoTokenizer,
                            messages::Vector{Dict{String, String}};
                            tokenize=true,
                            add_generation_prompt=true,
                            kwargs...)
    res = tokenizer.py_tokenizer.apply_chat_template(messages; tokenize, add_generation_prompt, kwargs...)
    if tokenize
        return pyconvert(Vector{Int}, res)
    else
        return pyconvert(String, res)
    end
end
apply_chat_template(tokenizer::Any, args...; kwargs...) = @error "apply_chat_template only supported for HuggingFaceTransformers.AutoTokenizer."

"""
    generate(model::CausalLM, prefix_tokens::Vector{Int64}; kw...) -> Vector{Array}
    generate(model::CausalLM, prompt::String, tokenizer; kw...) -> String

Generate text from a model. 
"""
generate(model::CausalLM, prefix_tokens::Vector{Int64}; kw...) = pyconvert(Vector{Array},model.py_transformer.generate(Torch.tensor(prefix_tokens).unsqueeze(0); kw...))[1]
generate(model::CausalLM, prompt::String, tokenizer::AutoTokenizer; kw...) = decode(tokenizer, pyconvert(Vector{Array},model.py_transformer.generate(Torch.tensor(encode(tokenizer, prompt)).unsqueeze(0); kw...))[1])


"""
    encode(tokenizer::Tokenizer, text::String, kwargs...) -> Vector{Int}

Encode a single text string into tokens and their corresponding IDs.
"""
function encode(tokenizer::AutoTokenizer, text::String; kwargs...)
    output = tokenizer.py_tokenizer.encode(text; kwargs...)
    ids = pyconvert(Vector{Int}, output)
    return ids
end

"""
    decode(tokenizer::Tokenizer, ids::Vector{Int}, skip_special_tokens = false) -> String

Decode a sequence of token IDs back into text.
"""
function decode(tokenizer::AutoTokenizer, args...; kwargs...)
    return pyconvert(String, tokenizer.py_tokenizer.decode(args...; kwargs...))
end

(model::HuggingFaceModel)(input::Py, args...; kw...) = model.py_transformer(input, args...; kw...)
(model::HuggingFaceModel)(input::Vector{Int64}; kw...) = model.py_transformer(Torch.tensor(input).unsqueeze(0); kw...)


"""
    tensor(x::Py) -> Array

Convert a PyTorch tensor to a Julia array, detaching the tensor.
"""
tensor(x::Py) = pyconvert(Array,x.detach())


