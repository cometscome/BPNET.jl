include("wrapper.jl")
include("generatebasis.jl")

mutable struct Generator
    outputname::String
    numkinds::Int64
    types::Vector{String}
    fingerprints::Vector{FingerPrint}
    filenames::Vector{String}
    numfiles::Int64
    isolated_energies::Vector{Float64}
    fingerprint_names::Vector{String}
    isgenerated::Bool
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


    isgenerated = false
    return Generator(outputname, numkinds, types, fingerprints, filenames, 0, isolated_energies, fingerprint_names, isgenerated)
end

function get_filename(g::Generator, i)
    return g.filenames[i]
end
export get_filename

function get_filenames(g::Generator)
    return g.filenames
end
export get_filenames

function make_descriptor(g::Generator)
    if isfile(g.outputname)
        @warn "$(g.outputname) exists. This is removed"
        rm(g.outputname)
    end

    filename = make_generatein(g)
    make_fingerprintfile(g)
    generate(filename)
    outputfile = g.outputname * ".ascii"
    g.isgenerated = true
    return outputfile
end
export make_descriptor

function make_fingerprintfile(g::Generator)
    for ifile = 1:g.numkinds
        filename = g.fingerprint_names[ifile]
        fingerprint = g.fingerprints[ifile]
        description = fingerprint.description

        fp = open(filename, "w")
        println(fp, "DESCR")
        println(fp, description)
        println(fp, "END DESCR")
        println(fp, "")
        println(fp, "ATOM ", fingerprint.atomtype)
        println(fp, "")
        println(fp, "ENV ", fingerprint.nenv)
        for i = 1:fingerprint.nenv
            println(fp, fingerprint.envtypes[i])
        end
        println(fp, "")
        println(fp, "RMIN ", fingerprint.rc_min)
        println(fp, "")
        println(fp, "BASIS type=", fingerprint.sftype)
        print_fingerprintinfo(fp, fingerprint)
        close(fp)
    end
end
export make_fingerprintfile

function print_fingerprintinfo(fp, fingerprint)
    if fingerprint.sftype == "Chebyshev"
        sfparam = fingerprint.sfparam
        radial_Rc = sfparam[1, 1]
        radial_N = Int64(sfparam[2, 1])
        angular_Rc = sfparam[3, 1]
        angular_N = Int64(sfparam[4, 1])

        print(fp, "radial_Rc = ", radial_Rc)
        print(fp, " radial_N = ", radial_N)
        print(fp, " angular_Rc = ", angular_Rc)
        print(fp, " angular_N = ", angular_N)
        println(fp, "\t")
    else
        error("fingerprint type $(fingerprint.sftype) is not supported")
    end
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





