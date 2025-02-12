struct BP_dataset{keys,num_of_types,num_of_structs,Tfp,Td,num_kinds}
    filename::String
    num_of_types::Int64
    num_of_structs::Int64
    type_names::Vector{String}
    E_atom::Vector{Float64}
    normalized::Bool
    E_scale::Float64
    E_shift::Float64
    natomtot::Int64
    E_avg::Float64
    E_min::Float64
    E_max::Float64
    has_setups::Bool
    fp::Tfp
    fingerprints::NamedTuple{keys,NTuple{num_of_types,FingerPrint}}
    #fingerprints::Vector{FingerPrint}
    headerposision::Int64
    datafilenames::Vector{String}
    fileused::Vector{Bool}
    datafile::Td
    fingerprint_parameters::Vector{Vector{FingerPrint_params}}
end

export BP_dataset


function get_inputdim(dataset::BP_dataset, name::String)
    return get_inputdim(dataset, Symbol(name))
end

function get_inputdim(dataset::BP_dataset, name::Symbol)
    return getfield(dataset.fingerprints, name).nsf
end
export get_inputdim

function Base.length(dataset::BP_dataset)
    return dataset.num_of_structs
end


function reload_data!(dataset::BP_dataset)
    dataset.fileused .= 0
    return nothing
end
export reload_data!


function BP_dataset_fromTOML(tomlfile)
    data = TOML.parsefile(tomlfile)
    display(data)
    filename = data["trainfile"]
    fp = open(filename, "r")
    num_of_types = parse(Int64, split(readline(fp))[1])
    num_of_structs = parse(Int64, split(readline(fp))[1])
    type_names = Vector{String}(undef, num_of_types)
    type_names .= collect(split(readline(fp)))
    @assert type_names == data["atomtypes"] "atomtypes should be $type_names"
    println(type_names)
    keys = Tuple(Symbol.(type_names))
    maxenergy = data["maxenergy"]



    E_atom = zeros(Float64, num_of_types)
    E_atom .= parse.(Float64, split(readline(fp)))

    u = split(readline(fp))[1]
    normalized = ifelse(u == "T", true, false)

    u = split(readline(fp))[1]
    E_scale = parse(Float64, u)
    u = split(readline(fp))[1]
    E_shift = parse(Float64, u)


    u = split(readline(fp))[1]
    natomtot = parse(Int64, u)
    E_avg, E_min, E_max = parse.(Float64, split(readline(fp)))
    u = split(readline(fp))[1]
    has_setups = ifelse(u == "T", true, false)


    fingerprints = Vector{FingerPrint}(undef, num_of_types)

    for jtype = 1:num_of_types
        itype = parse(Int64, split(readline(fp))[1])
        #println(itype)
        description = readline(fp)
        #println(description)
        atomtype = readline(fp)
        nenv = parse(Int64, split(readline(fp))[1])
        #println(nenv)
        envtypes = Vector{String}(undef, nenv)
        for k = 1:nenv
            u = split(readline(fp))[1]
            #println(u)
            envtypes[k] = u
        end
        #println(envtypes)
        rc_min = parse(Float64, split(readline(fp))[1])
        rc_max = parse(Float64, split(readline(fp))[1])
        #println((rc_min, rc_max))
        sftype = readline(fp)
        #println(sftype)
        nsf = parse(Int64, split(readline(fp))[1])
        nsfparam = parse(Int64, split(readline(fp))[1])
        #println((nsf, nsfparam))
        sf = parse.(Int64, split(readline(fp)))

        #println(sf)
        sfparam = zeros(Float64, nsfparam, nsf)
        sfparam[:] .= parse.(Float64, split(readline(fp)))
        #(sfparam)


        sfenv = zeros(Int64, 2, nsf)
        sfenv[:] .= parse.(Int64, split(readline(fp)))
        #display(sfenv)
        neval = parse(Int64, split(readline(fp))[1])

        sfval_min = zeros(Float64, nsf)
        sfval_min .= parse.(Float64, split(readline(fp)))
        sfval_max = zero(sfval_min)
        sfval_max .= parse.(Float64, split(readline(fp)))
        sfval_avg = zero(sfval_min)
        sfval_avg .= parse.(Float64, split(readline(fp)))
        sfval_cov = zero(sfval_min)
        sfval_cov .= parse.(Float64, split(readline(fp)))
        #display(sfval_min)
        #display(sfval_max)
        #display(sfval_avg)
        #display(sfval_cov)


        #println(readline(fp))

        fingerprints[itype] = FingerPrint(itype, description, atomtype,
            nenv,
            envtypes,
            rc_min,
            rc_max,
            sftype,
            nsf,
            nsfparam,
            sf,
            sfparam,
            sfenv,
            neval,
            sfval_min,
            sfval_max,
            sfval_avg,
            sfval_cov
        )
        #error("dd")
    end

    headerposision = position(fp)
    #println("Position: ", pos)
    fileheader = filename
    datafilenames = Vector{String}(undef, num_of_structs)
    for istruc = 1:num_of_structs
        datafilenames[istruc] = fileheader * "_data" * lpad(istruc, 7, '0') * ".jld2"
    end

    E_max = min(E_max, maxenergy)

    E_scale = 2.0 / (E_max - E_min)
    E_shift = 0.5 * (E_max + E_min)

    fileused = zeros(Int64, num_of_structs)

    fingerprintstuple = NamedTuple{keys}(fingerprints)


    fingerprint_parameters_set = Vector{Vector{FingerPrint_params}}(undef, length(type_names))
    if data["numbasiskinds"] != 1
        for itype = 1:length(type_names)
            fingerprint_i = getfield(fingerprintstuple, keys[itype])
            fingerprint_parameters_set[itype] = get_multifingerprints_info(fingerprint_i)
        end
        println(fingerprint_parameters_set)
    else
        for itype = 1:length(type_names)
            fingerprint_i = getfield(fingerprintstuple, keys[itype])
            inputdim = fingerprint_i.nsf
            fingerprint_parameters_set[itype] = get_singlefingerprints_info(fingerprint_i, inputdim)
        end
    end


    numbasiskinds = data["numbasiskinds"]
    if numbasiskinds == 1
        num_of_structs2 = writefulldata_to_jld2(fp, headerposision, num_of_structs, filename,
            type_names, E_shift, E_scale, datafilenames, fingerprintstuple, data["normalize"]
        )
        num_of_structs = num_of_structs2
    else
        num_of_structs2 = writefulldata_to_jld2_multi(data, fp, headerposision, num_of_structs, filename,
            type_names, E_shift, E_scale, datafilenames, fingerprintstuple
        )
        num_of_structs = num_of_structs2
    end

    dataffile = jldopen(fileheader * ".jld2", "r")

    dataset = BP_dataset{keys,num_of_types,num_of_structs,typeof(fp),typeof(dataffile),numbasiskinds}(
        filename,
        num_of_types, #::Int64
        num_of_structs, #::Int64
        type_names, #::Vector{String}
        E_atom, #::Vector{Float64}
        normalized, #::Bool
        E_scale, #::Float64
        E_shift, #::Float64
        natomtot, #::Int64
        E_avg, #::Float64
        E_min, #::Float64
        E_max, #::Float64
        has_setups, #::Bool
        fp, #::IOStream
        fingerprintstuple,
        #        NamedTuple{keys}(fingerprints),
        headerposision,
        datafilenames,
        fileused,
        dataffile,
        fingerprint_parameters_set
    )

    println("------------------------------------------------")
    println("dataset: $filename ")
    println("num. of data: $num_of_structs")
    println("------------------------------------------------")

    return dataset, data

end
export BP_dataset_fromTOML


function get_coefficients(dataset::BP_dataset{keys,num_of_types,num_of_structs}, istruct::Integer, fp) where {keys,num_of_types,num_of_structs}
    #@assert istruct <= num_of_structs "size of the dataset $(num_of_structs) is smaller than the index $istruct"
    dataset.fileused[istruct] = true
    energy = 0.0
    coefficients = Vector{Matrix{Float64}}(undef, num_of_types)
    natoms = 0
    #jldopen(dataset.datafilenames[istruct], "r") do file
    energy = fp["$istruct"]["energy"]
    coefficients .= fp["$istruct"]["coefficients"]
    natoms = fp["$istruct"]["natoms"]
    #println((energy[1], coefficients, natoms[1]))
    #end
    #@load dataset.datafilenames[istruct] energy coefficients natoms
    #println(typeof(coefficients))
    #error("dd")
    return energy, coefficients, natoms
end
export get_coefficients

function get_unusedindex(dataset::BP_dataset)
    return get_unusedindex_arr(dataset::BP_dataset, dataset.fileused)
end

function get_unusedindex(dataset::BP_dataset, dataindices)
    arr = view(dataset.fileused, dataindices)
    #@code_warntype get_unusedindex_arr(dataset::BP_dataset, arr)
    #error("dg")
    return get_unusedindex_arr(dataset::BP_dataset, arr)
end

function get_unusedindex_arr(dataset::BP_dataset, arr)
    arr = dataset.fileused

    count_zeros = count(==(0), arr)

    if count_zeros == 0
        #@warn "no unused data. reload_data! should be perfomed."
        s = 0
        #return 0
    else
        zero_index = findnext(==(0), arr, 1)
        for i in 2:rand(1:count_zeros)
            zero_index = findnext(==(0), arr, zero_index + 1)
        end
        s = ifelse(zero_index == nothing, 0, zero_index)
        #zero_index
        #return zero_index
    end
    return s
end


