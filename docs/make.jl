using HuggingFaceTransformers
using Documenter

DocMeta.setdocmeta!(HuggingFaceTransformers, :DocTestSetup, :(using HuggingFaceTransformers); recursive=true)

makedocs(;
    modules=[HuggingFaceTransformers],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="HuggingFaceTransformers.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/HuggingFaceTransformers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/HuggingFaceTransformers.jl",
    devbranch="main",
)
