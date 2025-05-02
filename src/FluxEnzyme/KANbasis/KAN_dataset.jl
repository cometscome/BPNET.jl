
mutable struct KAN_dataset
    #xsfdataset::XSFdataset
    numfile::Int64
    xsffilenames::Vector{String}
    kind_list::Vector{String}
    numkinds::Int64
    xsfdata::Vector{XSFdata}
    hasdata::Vector{Bool}
    energies::Vector{Float64}
    numatoms::Vector{Int64}
    isrescaled::Bool
    isolated_energies::Vector{Float64}

    function KAN_dataset(kind_list, isolated_energies::Vector{Float64})
        numfile = 0
        xsffilenames = String[]
        xsfdata = XSFdata[]
        hasdata = Bool[]
        energies = Float64[]
        numatoms = Int64[]
        #xsfdataset = XSFdataset()
        numkinds = length(kind_list)
        isrescaled = false
        #return new(xsfdataset, kind_list, numkinds)
        return new(numfile, xsffilenames, kind_list, numkinds, xsfdata,
            hasdata, energies, numatoms, isrescaled,
            isolated_energies)
    end

    function KAN_dataset(kind_list, filenames, isolated_energies::Vector{Float64})
        xsffilenames = String[]
        numfile = 0
        #xsfdataset = XSFdataset()
        for filename in filenames
            numfile += 1
            push!(xsffilenames, filename)
            #push!(xsfdataset, filename)
        end
        xsfdata = Vector{XSFdata}(undef, numfile)
        hasdata = zeros(numfile)
        energies = zeros(numfile)
        numatoms = zeros(Int64, numfile)
        numkinds = length(kind_list)
        isrescaled = false
        return new(numfile, xsffilenames, kind_list, numkinds, xsfdata,
            hasdata, energies, numatoms, isrescaled,
            isolated_energies)
        #return new(xsfdataset, kind_list, numkinds)
    end
end
export KAN_dataset

function Base.length(dataset::KAN_dataset)
    return dataset.numfile
end



function Base.push!(xsf::KAN_dataset, filename::String)
    xsf.numfile += 1
    resize!(xsf.xsfdata, xsf.numfile)
    resize!(xsf.hasdata, xsf.numfile)
    resize!(xsf.energies, xsf.numfile)
    resize!(xsf.numatoms, xsf.numfile)
    xsf.hasdata[xsf.numfile] = 0
    push!(xsf.xsffilenames, filename)
end

function rescale_energies!(dataset::KAN_dataset, E_scale, E_shift)
    numfile = length(dataset)
    for i = 1:numfile
        if dataset.hasdata[i]
        else
            filename = dataset.xsffilenames[i]
            dataset.xsfdata[i] = XSFdata(filename)
            dataset.hasdata[i] = true
            nums = get_number_each(dataset.xsfdata[i], kind_list)
            isolated_energies_total = sum(nums .* dataset.isolated_energies)
            dataset.energies[i] = get_energy(dataset.xsfdata[i]) - isolated_energies_total
            dataset.numatoms[i] = get_number(dataset.xsfdata[i])
        end

        natoms = dataset.numatoms[i]
        dataset.energies[i] = E_scale * (dataset.energies[i] - natoms * E_shift)
    end
    dataset.isrescaled = true
end
export rescale_energies!

function remove_highenergy_structures!(dataset::KAN_dataset, E_max=1.001)
    numfile = length(dataset)
    count = 0
    xsffilenames = String[]
    xsfdata = XSFdata[]
    hasdata = Bool[]
    energies = Float64[]
    numatoms = Int64[]


    for i = 1:numfile
        if dataset.hasdata[i]
        else
            filename = dataset.xsffilenames[i]
            dataset.xsfdata[i] = XSFdata(filename)
            dataset.hasdata[i] = true
            nums = get_number_each(dataset.xsfdata[i], kind_list)
            isolated_energies_total = sum(nums .* dataset.isolated_energies)
            dataset.energies[i] = get_energy(dataset.xsfdata[i]) - isolated_energies_total
            dataset.numatoms[i] = get_number(dataset.xsfdata[i])
        end
        energy_peratom = dataset.energies[i] / dataset.numatoms[i]
        if energy_peratom < E_max
            count += 1
            push!(xsffilenames, dataset.xsffilenames[i])
            push!(xsfdata, dataset.xsfdata[i])
            push!(hasdata, 1)
            push!(energies, dataset.energies[i])
            push!(numatoms, dataset.numatoms[i])
        end
    end

    if count < dataset.numfile
        println("$( dataset.numfile -count) high-energy structures are skipped")
    end

    dataset.xsffilenames = xsffilenames
    dataset.xsfdata = xsfdata
    dataset.hasdata = hasdata
    dataset.energies = energies
    dataset.numatoms = numatoms
    dataset.numfile = count
end
export remove_highenergy_structures!

function get_partialsumfilter(kind_list, xsfs::AbstractVector{XSFdata})
    numkinds = length(kind_list)
    numfiles = length(xsfs)
    partialsumfilter = Vector{Matrix{Int64}}(undef, numkinds)
    position = zeros(1, numfiles)

    for ikind = 1:numkinds
        partialsumfilter[ikind] = Matrix{Int64}(undef, 0, numfiles)
    end

    for ifile = 1:numfiles
        xsf_i = xsfs[ifile]
        position[ifile] = 1

        for ikind = 1:numkinds
            numatoms_ikind = get_number_kind(xsf_i, kind_list[ikind])

            for iatom = 1:numatoms_ikind
                partialsumfilter[ikind] = vcat(partialsumfilter[ikind], position)
            end
        end

        position[ifile] = 0
    end
    return partialsumfilter
end
export get_partialsumfilter

function get_energies(dataset::KAN_dataset)
    numfile = length(dataset)
    energies = zeros(numfile)
    for i = 1:numfile
        filename = dataset.xsffilenames[i]

        if dataset.hasdata[i]
        else
            dataset.xsfdata[i] = XSFdata(filename)
            dataset.hasdata[i] = true
            nums = get_number_each(dataset.xsfdata[i], kind_list)
            isolated_energies_total = sum(nums .* dataset.isolated_energies)
            dataset.energies[i] = get_energy(dataset.xsfdata[i]) - isolated_energies_total
            dataset.numatoms[i] = get_number(dataset.xsfdata[i])
        end
        energies[i] = dataset.energies[i]#get_energy(dataset.xsfdata[i])
    end
    return energies
end
export get_energies

function get_energies_and_numatoms(dataset::KAN_dataset)
    numfile = length(dataset)
    energies = zeros(numfile)
    numatoms = zeros(numfile)
    for i = 1:numfile
        filename = dataset.xsffilenames[i]

        if dataset.hasdata[i]
        else
            dataset.xsfdata[i] = XSFdata(filename)
            dataset.hasdata[i] = true
            nums = get_number_each(dataset.xsfdata[i], dataset.kind_list)
            isolated_energies_total = sum(nums .* dataset.isolated_energies)
            dataset.energies[i] = get_energy(dataset.xsfdata[i]) - isolated_energies_total

            dataset.numatoms[i] = get_number(dataset.xsfdata[i])
            #dataset.energies[i] = get_energy(dataset.xsfdata[i])

        end
        energies[i] = dataset.energies[i]#get_energy(dataset.xsfdata[i])
        numatoms[i] = dataset.numatoms[i]#get_number(dataset.xsfdata[i])
    end
    return energies, numatoms
end
export get_energies, get_energies_and_numatoms



function Base.getindex(dataset::KAN_dataset,
    I::AbstractVector)
    num = length(I)
    energy_batch = zeros(Float64, 1, num)
    #xsfs = XSFdata[]

    for (count, i) in enumerate(I)
        filename = dataset.xsffilenames[i]
        #push!(xsfs, XSFdata(filename))

        if dataset.hasdata[i]
        else
            dataset.xsfdata[i] = XSFdata(filename)
            dataset.hasdata[i] = true
            dataset.energies[i] = get_energy(dataset.xsfdata[i])
        end
        energy_batch[count] = dataset.energies[i]#get_energy(dataset.xsfdata[i])

    end
    xsfs = view(dataset.xsfdata, I)
    pos = get_partialsumfilter(dataset.kind_list, xsfs)
    #println(typeof(pos))
    return pos, xsfs, energy_batch
end