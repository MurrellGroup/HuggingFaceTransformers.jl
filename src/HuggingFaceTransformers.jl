"""
    HuggingFaceTokenizers

A Julia wrapper around HuggingFace's Tokenizers Python library.

See https://huggingface.co/docs/tokenizers/en/index for official documentation.
"""
module HuggingFaceTransformers
using PythonCall

const Torch = PythonCall.pynew()
const Transformers = PythonCall.pynew()
const Einops = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(Torch, pyimport("torch"))
    PythonCall.pycopy!(Transformers, pyimport("transformers"))
    PythonCall.pycopy!(Einops, pyimport("einops"))
end

include("transformers.jl")
export Model
export CausalLM
export AutoTokenizer
export from_pretrained
export generate
export encode
export decode
export apply_chat_template
export tensor

end
