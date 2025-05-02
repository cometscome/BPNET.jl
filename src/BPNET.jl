module BPNET
using Flux
using Random
using Requires
using JLD2
using MLUtils



include("XSFreader.jl")
using .XSFreader
export XSFdata
export get_atoms_inside_the_sphere
export get_atoms_inside_the_sphere!
export splitR_xsf, get_energy, get_number_each




include("fingerprints.jl")
include("dataloader/dataloader_xsf.jl")

#include("generate/generate.jl")



include("train/train.jl")
include("predict/predict.jl")
include("train/Fullmemory.jl")
include("train/BPnet.jl")
include("train/training.jl")
include("Luxmodel/luxmodel.jl")
include("Luxmodel/fluxmodel.jl")



include("FluxEnzyme/BPNET_FluxEnzyme.jl")
import .BPNET_FluxEnzyme: Generator,
    KAN_dataset,
    BNnetwork,
    KAN_Layer,
    ChebyshevLayer,
    KANbasis,
    get_numbasis,
    IdentityTupleLayer,
    get_energies,
    get_energies_and_numatoms,
    get_isolated_energies,
    rescale_energies!,
    remove_highenergy_structures!

export Generator,
    KAN_dataset,
    BNnetwork,
    KAN_Layer,
    ChebyshevLayer,
    KANbasis,
    get_numbasis,
    IdentityTupleLayer,
    get_energies,
    get_energies_and_numatoms,
    get_isolated_energies,
    rescale_energies!,
    remove_highenergy_structures!



function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin end
end
# Write your package code here.

end
