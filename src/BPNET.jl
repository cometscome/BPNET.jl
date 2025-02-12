module BPNET
include("generate/generate.jl")
include("train/train.jl")
include("predict/predict.jl")

using Flux
using Requires


function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin end
end
# Write your package code here.

end
