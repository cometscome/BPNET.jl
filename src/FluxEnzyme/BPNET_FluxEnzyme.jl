module BPNET_FluxEnzyme
using Flux

#include("fingerprints.jl")
import ..BPNET: FingerPrint
include("generate.jl")


using ..XSFreader


include("KANbasis/KANbasis.jl")
include("KANbasis/KAN_dataset.jl")


end