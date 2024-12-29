"""
    HuggingFaceTokenizers

A Julia wrapper around HuggingFace's Tokenizers Python library.

See https://huggingface.co/docs/tokenizers/en/index for official documentation.
"""
module HuggingFaceTransformers
using PythonCall
using HuggingFaceTokenizers

const Torch = PythonCall.pynew()
const Transformers = PythonCall.pynew()
const AutoModelForCausalLM = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(Torch, pyimport("torch"))
    PythonCall.pycopy!(Transformers, pyimport("transformers"))
    PythonCall.pycopy!(AutoModelForCausalLM, Transformers.AutoModelForCausalLM)
end

include("transformers.jl")
export Transformer
export AutoTokenizer
export from_pretrained
export generate
export encode
export decode
export apply_chat_template

end
