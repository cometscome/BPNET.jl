module BPNET
using Flux
using Random
using Requires
using JLD2
using MLUtils


include("fingerprints.jl")
include("generate/generate.jl")
include("train/train.jl")
include("predict/predict.jl")
include("train/Fullmemory.jl")
include("train/BPnet.jl")
include("train/training.jl")



function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin end
end
# Write your package code here.

end
