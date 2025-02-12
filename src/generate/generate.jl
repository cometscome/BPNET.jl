mutable struct Generator
    outputname::String
    numkinds::Int64
    types::Vector{String}
    fingerprints::Vector{FingerPrint}
    filenames::Vector{String}
    numfiles::Int64
    isolated_energies::Vector{Float64}
    fingerprint_names::Vector{String}
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

    if haskey(kwargs, :outputname)
        outputname = kwargs[:outputname]
    else
        outputname = prod(types) * ".train"
    end


    if haskey(kwargs, :fingerprint_names)
        fingerprint_names = kwargs[:fingerprint_names]
    else
        fingerprint_names = Vector{String}(undef, numkinds)
        for i = 1:numkinds
            fingerprint_names[i] = types[i] * ".fingerprint.stp"
        end
    end



    return Generator(outputname, numkinds, types, fingerprints, filenames, 0, isolated_energies, fingerprint_names)
end

function make_generatein(g::Generator; filename="generate.in")
    fp = open(filename, "w")
    println(fp, "OUTPUT $(g.outputname)")
    println(fp, "\t")
    println(fp, "TYPES")
    println(fp, g.numkinds)
    for i = 1:g.numkinds
        println(fp, g.types[i], " ", g.isolated_energies[i], "  | eV")
    end
    println(fp, "\t")
    println(fp, "SETUPS")
    for i = 1:g.numkinds
        println(fp, g.types[i], " ", g.fingerprint_names[i])
    end
    println(fp, "\t")
    println(fp, "FILES")
    println(fp, g.numfiles)
    for i = 1:g.numfiles
        println(fp, g.filenames[i])
    end
    close(fp)
    return filename
end
export make_generatein

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





