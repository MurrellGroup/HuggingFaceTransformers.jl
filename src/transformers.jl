import HuggingFaceTokenizers: from_pretrained, encode, decode

"""
    Tokenizer

A wrapper around a Python tokenizer.
"""
struct Transformer
    py_transformer::Py
end

struct AutoTokenizer
    py_tokenizer::Py
end

"""
    model_from_pretrained(model_path::String) -> Model

Load a pretrained model. `m` can be a model name or a path to a local model, or a HuggingFace URL.
"""
from_pretrained(::Type{Transformer}, m::String) = Transformer(AutoModelForCausalLM.from_pretrained(m))

"""
    from_pretrained(model_path::String) -> Tokenizer

Load a pretrained AutoTokenizer. `m` can be a model name or a path to a local model, or a HuggingFace URL.
Compared to HuggingFaceTokenizers.jl, this can use eg. chat templates.
"""
from_pretrained(::Type{AutoTokenizer}, m::String) = AutoTokenizer(Transformers.AutoTokenizer.from_pretrained(m))

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
apply_chat_template(tokenizer::HuggingFaceTokenizers.Tokenizer, args...; kwargs...) = @error "apply_chat_template not supported for HuggingFaceTokenizers.Tokenizer. Use a HuggingFaceTransformers.AutoTokenizer instead."

"""
    generate(model::Transformer, prefix_tokens::Vector{Int64}; kw...) -> Vector{Array}
    generate(model::Transformer, prompt::String, tokenizer; kw...) -> String

Generate text from a model. 
"""
generate(model::Transformer, prefix_tokens::Vector{Int64}; kw...) = pyconvert(Vector{Array},model.py_transformer.generate(Torch.tensor(prefix_tokens).unsqueeze(0); kw...))[1]
generate(model::Transformer, prompt::String, tokenizer::HuggingFaceTokenizers.Tokenizer; kw...) = decode(tokenizer, pyconvert(Vector{Array},model.py_transformer.generate(Torch.tensor(encode(tokenizer, prompt).ids).unsqueeze(0); kw...))[1])
generate(model::Transformer, prompt::String, tokenizer::AutoTokenizer; kw...) = decode(tokenizer, pyconvert(Vector{Array},model.py_transformer.generate(Torch.tensor(encode(tokenizer, prompt)).unsqueeze(0); kw...))[1])


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