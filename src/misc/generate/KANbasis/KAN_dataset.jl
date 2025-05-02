
mutable struct KAN_dataset
    #xsfdataset::XSFdataset
    numfile::Int64
    xsffilenames::Vector{String}
    kind_list::Vector{String}
    numkinds::Int64
    xsfdata::Vector{XSFdata}
    hasdata::Vector{Bool}

    function KAN_dataset(kind_list)
        numfile = 0
        xsffilenames = String[]
        xsfdata = XSFdata[]
        hasdata = Bool[]
        #xsfdataset = XSFdataset()
        numkinds = length(kind_list)
        #return new(xsfdataset, kind_list, numkinds)
        return new(numfile, xsffilenames, kind_list, numkinds, xsfdata, hasdata)
    end

    function KAN_dataset(kind_list, filenames)
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
        numkinds = length(kind_list)
        return new(numfile, xsffilenames, kind_list, numkinds, xsfdata, hasdata)
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
    xsf.hasdata[xsf.numfile] = 0
    push!(xsf.xsffilenames, filename)
end

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



function Base.getindex(dataset::KAN_dataset,
    I::AbstractVector)
    num = length(I)
    energy_batch = zeros(Float64, 1, num)
    #xsfs = XSFdata[]

    for (count, i) in enumerate(I)
        filename = dataset.xsffilenames[i]
        #push!(xsfs, XSFdata(filename))
        dataset.hasdata[i] = true
        dataset.xsfdata[i] = XSFdata(filename)
        energy_batch[count] = get_energy(dataset.xsfdata[i])
    end
    xsfs = view(dataset.xsfdata, I)
    pos = get_partialsumfilter(dataset.kind_list, xsfs)
    #println(typeof(pos))
    return pos, xsfs, energy_batch
end