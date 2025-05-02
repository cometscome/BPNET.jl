

abstract type AbstractbasisLayer end

function write_descriptor(xsf_i::XSFdata, m::AbstractbasisLayer, filename,
    kind_list; Rmax=8.0)
    error("basis type $(typeof(m)) is not supported in write_descriptor")
end
export write_descriptor

function generate_descriptors(m::AbstractbasisLayer, xsffilelist, kind_list; Rmax=8.0, footer="")
    numdata = length(xsffilelist)
    for (i, filename) in enumerate(xsffilelist)
        xsf_i = XSFdata(filename)
        filename_jld = filename[1:end-4] * footer #* ".jld2"
        #println(filename_jld)
        println("$i/$(numdata)")
        write_descriptor(xsf_i, m, filename_jld, kind_list; Rmax)
    end
end
export generate_descriptors

include("chebyshevbasis/chebyshev.jl")
include("chebyshevbasis/ChebyshevLayer.jl")
include("chebyshevbasis/KACbasis_radial.jl")
include("chebyshevbasis/KACbasis_angular.jl")



include("KANbasis/KAN_chebyshev.jl")
include("KANbasis/KAN_chebyshev_radial.jl")
include("KANbasis/KAN_chebyshev_angular.jl")
include("KANbasis/KAN_composit_layer.jl")
include("KANbasis/KAN_layer.jl")


include("KANbasis/KAN_dataset.jl")

