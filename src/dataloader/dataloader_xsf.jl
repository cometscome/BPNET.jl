

mutable struct XSFdataset
    numfile::Int64
    xsffilenames::Vector{String}
    xsfdata::Vector{XSFdata}
end

function XSFdataset()
    numfile = 0
    xsffilenames = String[]
    xsfdata = XSFdata[]
    return XSFdataset(numfile, xsffilenames, xsfdata)
end

function Base.push!(xsf::XSFdataset, filename::String)
    xsf.numfile += 1
    xsfdata = XSFdata(filename)
    push!(xsf.xsffilenames, filename)
    push!(xsf.xsfdata, xsfdata)
end

function Base.getindex(xsf::XSFdataset, i::Int64)
    #filename = xsf.xsffilenames[i]
    xsf_i = xsf.xsfdata[i]
    return xsf_i
    #numatoms = get_number(xsf_i)
    #for iatom = 1:numatoms
    #end
end
