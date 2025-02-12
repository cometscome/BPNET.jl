mutable struct Generator
    numkinds::Int64
    types::Vector{String}
    fingerprints::Vector{FingerPrint}
    filenames::Vector{String}
    numfiles::Int64
    isolated_energies::Vector{Float64}
end
export Generator

function Generator(types; kwargs...)
    numkinds = length(types)
    #types = Vector{String}(undef, numkinds)
    fingerprints = Vector{FingerPrint}(undef, numkinds)
    filenames = String[]
    if haskey(kwargs, :isolated_energies)
        isolated_energies = kwargs[:isolated_energies]
    else
        isolated_energies = zeros(numkinds)
    end

    return Generator(numkinds, types, fingerprints, filenames, 0, isolated_energies)
end

function Base.push!(g::Generator, f::FingerPrint)
    @assert g.types == f.envtypes "atomic type is wrong in fingerprints!"
    g.fingerprints[f.itype] = deepcopy(f)
end

function adddata!(g::Generator, data::String)
    push!(g.filenames, data)
    g.numfiles += 1
end

function adddata!(g::Generator, data::Vector{String})
    for datai in data
        adddata!(g, datai)
    end
end
export adddata!

function set_numfiles!(g::Generator, numfiles)
    g.numfiles = numfiles
end
export set_numfiles!





