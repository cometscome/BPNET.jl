abstract type KAN_layer{T} end
abstract type KAN_Chebyshev_layer{T} <: KAN_layer{T} end

function get_numbasis(x::KAN_Chebyshev_layer)
    return x.numbasis
end
export get_numbasis

function Base.display(x::KAN_Chebyshev_layer)
    Rmin = x.basis.Rmin
    Rc = x.basis.Rc
    polynomial_order = x.basis.polynomial_order
    println("Chebyshev descriptor $(typeof(x))")
    println("Rmin = $(Rmin)")
    println("Rc = $(Rc)")
    println("polynomial order = $(polynomial_order)")
    println("num. of basis = $(x.numbasis)")
end

function (x::KAN_Chebyshev_layer{T})(xsf::XSFdata) where {T}
    cn = Dict{String,Matrix{T}}()#Vector{Matrix{T}}(undef, x.numkinds)
    for ikind = 1:x.numkinds
        cn[x.kind_list[ikind]] = Matrix{T}(undef, x.numbasis, 0)
    end
    for ith_atom = 1:xsf.numatoms
        R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(xsf, ith_atom, x.basis.Rc)
        cni = x(R_js, atomkinds_j, x.kind_list, x.colors)
        cn[atomkind_i] = hcat(cn[atomkind_i], cni)
    end
    return Tuple(values(cn))
end

function (x::KAN_Chebyshev_layer)(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors) where {T<:Real}
    m = x.basis
    y = m(R_js, kinds_j, kind_list, colors)

    return (y .- x.bn) ./ x.an
end



function (x::KAN_Chebyshev_layer{T})(xsfs::Vector{XSFdata}) where {T}
    numfiles = length(xsfs)
    numkinds = x.numkinds

    cnset = Vector{Matrix{T}}(undef, numkinds)
    for ikind = 1:numkinds
        cnset[ikind] = Matrix{T}(undef, x.numbasis, 0)
    end

    for ifile = 1:numfiles
        xsf_i = xsfs[ifile]
        cn = x(xsf_i)
        for ikind = 1:x.numkinds
            cn_i = cn[ikind]
            cnset[ikind] = hcat(cnset[ikind], cn_i)
        end
    end
    return cnset
end


