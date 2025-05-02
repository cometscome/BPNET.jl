using InteractiveUtils
using Flux

struct ElementwiseMul
end

(m::ElementwiseMul)(e, P) = e .* P
Flux.@layer ElementwiseMul #trainable = ()
export ElementwiseMul

struct ReduceLayer end
function (::ReduceLayer)(de)
    sum(reduce((x, y) -> x .* y, de))
end
Flux.@layer ReduceLayer
export ReduceLayer

#── 1) Kanbasis を適用し，pos をそのままタプルで返す層
struct KanBasisLayer
    kanbasis  # 関数オブジェクト
end
function (m::KanBasisLayer)(x::Tuple{Vector{Matrix{Float64}},
    Vector{XSFdata_staticfields}})
    Rs, xsf = x
    c = m.kanbasis(Rs, xsf)
    return c
end
Flux.@layer KanBasisLayer trainable = (kanbasis,)
export KanBasisLayer

#── 2) pmodel を適用し，pos をそのままタプルで返す層
struct PModelLayer
    pmodel
end
function (m::PModelLayer)(x)
    c, pos = x
    d = m.pmodel(c)
    return (d, pos)
end
Flux.@layer PModelLayer trainable = (pmodel,)
export PModelLayer

#── 4) 配列のタプルを受け取り，全要素の要素積を返す層
function array_prod(xs)
    prod = xs[1]
    for x in xs[2:end]
        prod .= prod .* x
    end
    return prod
end
struct ProdReduceLayer end
function (m::ProdReduceLayer)(de)
    return array_prod(de)
end
Flux.@layer ProdReduceLayer
export ProdReduceLayer

struct SumLayer end
function (m::SumLayer)(x)
    return sum(x)
end
Flux.@layer SumLayer
export SumLayer

struct ElemMulLayer end
function (m::ElemMulLayer)(x)
    d, pos = x
    return d .* pos
end
Flux.@layer ElemMulLayer
export ElemMulLayer


struct IdentityLayer end
(::IdentityLayer)(x) = x
export IdentityLayer
Flux.@layer IdentityLayer #trainable = ()

struct IdentityTupleLayer end
(::IdentityTupleLayer)(x...) = x
export IdentityTupleLayer
Flux.@layer IdentityTupleLayer #trainable = ()

struct TupleWrap end
(::TupleWrap)(x) = Tuple(x)
Flux.@layer TupleWrap #trainable = ()
export TupleWrap

struct BNnetwork{TK,TP}
    kanbasis::TK
    pmodel::TP
end
Flux.@layer BNnetwork trainable = (kanbasis, pmodel)
export BNnetwork

function (m::BNnetwork)(Rs, xsfstatics, pos)
    return calc_BNenergy(Rs, xsfstatics, pos, m.pmodel, m.kanbasis)
end

function (m::BNnetwork)(Rs, xsfstatics, pos, energies)
    ytilde = calc_BNenergy(Rs, xsfstatics, pos, m.pmodel, m.kanbasis)
    #return Flux.mse(ytilde, energies)


    d = 0.0
    n = length(energies)
    for i = 1:n
        d += (ytilde[i] - energies[i])^2
    end
    return d / n
    #return Flux.mse(ytilde, energies)

end


function calc_BNenergy(Rs, xsfstatics, pos, pmodel, kanbasis)
    c = kanbasis(Rs, xsfstatics)
    d = pmodel(c)
    de = d .* pos
    #println(de)
    result = reduce((x, y) -> x .* y, de)
    #return sum(result)
    return result
    #return sum(sum.(de))
end



struct KANbasis{numkinds,keys,T}
    kanlayers::NamedTuple{keys,T}#
    kind_list::Vector{String}

    function KANbasis(kanlayers::NamedTuple{keys_in,T}, kind_list) where {keys_in,T}
        numkinds = length(kanlayers)
        return new{numkinds,keys_in,T}(kanlayers, kind_list)
    end

    function KANbasis(layers::NamedTuple{keys_in,T}) where {keys_in,T}
        numkinds = length(layers)
        kind_list = String.(keys(layers))
        return new{numkinds,keys_in,T}(layers, collect(kind_list))
        #return new{numkinds,keys_in,T}(layers, Tuple(kind_list))
    end
end
export KANbasis

function get_numbasis(x::KANbasis{numkinds}, atomkind_i) where {numkinds}
    kanlayer = x.kanlayers[Symbol(atomkind_i)]
    numbasis = get_numbasis(kanlayer)
    return numbasis
end

Flux.@layer KANbasis trainable = kanlayers

function KANbasis_forward_matrix(numkinds, kinds, kanlayers, numatoms, R, cell, kind_list, R_i, R_js)
    ith_atom = 1
    atomkind_i_symbol = Symbol(kinds[ith_atom])
    kanlayer = kanlayers[atomkind_i_symbol]
    basis = kanlayer.basis

    #atomkind_i, index_i, atomkinds_j, indices_j, numneighbors = get_atoms_inside_the_sphere!(
    #    numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js)
    atomkinds_j = String[]
    indices_j = Int64[]
    atomkind_i, index_i, numneighbors = get_atoms_inside_the_sphere!(
        numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js, indices_j, atomkinds_j)



    cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
    #@code_warntype kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
    #error("dd")

    T = eltype(cni)
    #T = Float64
    cn = Vector{Matrix{T}}(undef, numkinds)
    for (ikind, kanlayer) in enumerate(kanlayers)
        numbasis = get_numbasis(kanlayer)
        cn[ikind] = zeros(numbasis, 0)
    end
    ikind_i = findfirst(x -> x == kinds[ith_atom], kind_list)
    cn[ikind_i] = hcat(cn[ikind_i], cni)

    for ith_atom = 2:numatoms
        atomkind_i_symbol = Symbol(kinds[ith_atom])
        kanlayer = kanlayers[atomkind_i_symbol]
        basis = kanlayer.basis
        atomkind_i, index_i, atomkinds_j, indices_j, numneighbors = get_atoms_inside_the_sphere!(
            numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js)

        cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
        ikind_i = findfirst(x -> x == kinds[ith_atom], kind_list)
        cn[ikind_i] = hcat(cn[ikind_i], cni)
        #cn[atomkind_i] = hcat(cn[atomkind_i], cni)
    end
    return cn #Tuple(cn)
end

function KANbasis_forward_matrix(numkinds, kinds, kanlayers, numatoms, R, cell, kind_list)
    ith_atom = 1
    atomkind_i_symbol = Symbol(kinds[ith_atom])
    kanlayer = kanlayers[atomkind_i_symbol]
    basis = kanlayer.basis

    #atomkind_i, index_i, atomkinds_j, indices_j, numneighbors = get_atoms_inside_the_sphere!(
    #    numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js)
    atomkinds_j = String[]
    indices_j = Int64[]
    nmax = 400
    R_js = zeros(3, nmax)
    R_i = zeros(3)

    atomkind_i, index_i, numneighbors = get_atoms_inside_the_sphere!(
        numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js, indices_j, atomkinds_j)



    cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
    #@code_warntype kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
    #error("dd")

    T = eltype(cni)
    #T = Float64
    cn = Vector{Matrix{T}}(undef, numkinds)
    for (ikind, kanlayer) in enumerate(kanlayers)
        numbasis = get_numbasis(kanlayer)
        cn[ikind] = zeros(numbasis, 0)
    end
    ikind_i = findfirst(x -> x == kinds[ith_atom], kind_list)
    cn[ikind_i] = hcat(cn[ikind_i], cni)

    for ith_atom = 2:numatoms
        atomkind_i_symbol = Symbol(kinds[ith_atom])
        kanlayer = kanlayers[atomkind_i_symbol]
        basis = kanlayer.basis
        atomkind_i, index_i, atomkinds_j, indices_j, numneighbors = get_atoms_inside_the_sphere!(
            numatoms, ith_atom, basis.Rc, R, kinds, cell, R_i, R_js)

        cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors, numneighbors)
        ikind_i = findfirst(x -> x == kinds[ith_atom], kind_list)
        cn[ikind_i] = hcat(cn[ikind_i], cni)
        #cn[atomkind_i] = hcat(cn[atomkind_i], cni)
    end
    return cn #Tuple(cn)
end

#=
function KANbasis_forward(numkinds, kinds, kanlayers, numatoms, R, cell, kind_list)
    ith_atom = 1
    atomkind_i_symbol = Symbol(kinds[ith_atom])
    kanlayer = kanlayers[atomkind_i_symbol]
    basis = kanlayer.basis

    R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(numatoms, ith_atom, basis.Rc, R, kinds, cell)
    #R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(xsf, ith_atom, basis.Rc)
    cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors)
    T = eltype(cni)
    cn = Dict{String,Matrix{T}}()#Vector{Matrix{T}}(undef, x.numkinds)
    for ikind = 1:numkinds
        atomkind_i_symbol = Symbol(kinds[ith_atom])
        kanlayer = kanlayers[atomkind_i_symbol]
        numbasis = get_numbasis(kanlayer)
        cn[kind_list[ikind]] = Matrix{T}(undef, numbasis, 0)
    end
    cn[atomkind_i] = hcat(cn[atomkind_i], cni)

    for ith_atom = 2:numatoms
        atomkind_i_symbol = Symbol(kinds[ith_atom])
        kanlayer = kanlayers[atomkind_i_symbol]
        basis = kanlayer.basis
        R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(numatoms, ith_atom, basis.Rc, R, kinds, cell)
        cni = kanlayer(R_js, atomkinds_j, kind_list, kanlayer.colors)
        cn[atomkind_i] = hcat(cn[atomkind_i], cni)
    end
    return Tuple(values(cn))
end
=#

function (x::KANbasis{numkinds})(kinds, numatoms, R, cell, R_i, R_js) where {numkinds}
    c = KANbasis_forward_matrix(numkinds, kinds, x.kanlayers, numatoms, R, cell, x.kind_list, R_i, R_js)
    #println(typeof(c))
    return ntuple(i -> c[i], numkinds)
end

function (x::KANbasis{numkinds})(R, xsfstatic::XSFdata_staticfields, R_i, R_js) where {numkinds}
    c = KANbasis_forward_matrix(numkinds, xsfstatic.kinds, x.kanlayers, xsfstatic.numatoms, R, xsfstatic.cell, x.kind_list, R_i, R_js)
    #println(typeof(c))
    return ntuple(i -> c[i], numkinds)
end

function (x::KANbasis{numkinds})(R, xsfstatic::XSFdata_staticfields) where {numkinds}
    c = KANbasis_forward_matrix(numkinds, xsfstatic.kinds, x.kanlayers, xsfstatic.numatoms, R, xsfstatic.cell, x.kind_list)
    #println(typeof(c))
    return ntuple(i -> c[i], numkinds)
end

function (x::KANbasis{numkinds})(Rs, vecxsfstatic::AbstractVector{XSFdata_staticfields}) where {numkinds}
    numfiles = length(vecxsfstatic)
    ifile = 1
    xsf_i = vecxsfstatic[ifile]
    Rs_i = Rs[ifile]

    cn = x(Rs_i, xsf_i)
    T = eltype(cn[1])

    cnset = Vector{Matrix{T}}(undef, numkinds)
    for ikind = 1:numkinds
        numbasis = get_numbasis(x, x.kind_list[ikind])
        cnset[ikind] = Matrix{T}(undef, numbasis, 0)
    end
    for ikind = 1:numkinds
        cn_i = cn[ikind]
        cnset[ikind] = hcat(cnset[ikind], cn_i)
    end

    for ifile = 2:numfiles
        xsf_i = vecxsfstatic[ifile]
        Rs_i = Rs[ifile]
        cn = x(Rs_i, xsf_i)

        for ikind = 1:numkinds
            cn_i = cn[ikind]
            cnset[ikind] = hcat(cnset[ikind], cn_i)
        end
    end
    return ntuple(i -> cnset[i], numkinds)
end

function (x::KANbasis{numkinds})(xsf::XSFdata, R_i, R_js) where {numkinds}
    c = KANbasis_forward_matrix(numkinds, xsf.kinds, x.kanlayers, xsf.numatoms, xsf.R, xsf.cell, x.kind_list, R_i, R_js)
    #println(typeof(c))
    return ntuple(i -> c[i], numkinds)
end

#=
function (x::KANbasis{numkinds})(xsf::XSFdata) where {numkinds}
    return KANbasis_forward(numkinds, xsf.kinds, x.kanlayers, xsf.numatoms, xsf.R, xsf.cell, x.kind_list)


    ith_atom = 1
    atomkind_i_symbol = Symbol(xsf.kinds[ith_atom])
    kanlayer = x.kanlayers[atomkind_i_symbol]
    basis = kanlayer.basis
    #@code_llvm get_atoms_inside_the_sphere(xsf, ith_atom, basis.Rc)
    #error("dd")
    R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(xsf, ith_atom, basis.Rc)
    cni = kanlayer(R_js, atomkinds_j, x.kind_list, kanlayer.colors)
    T = eltype(cni)
    cn = Dict{String,Matrix{T}}()#Vector{Matrix{T}}(undef, x.numkinds)
    for ikind = 1:numkinds
        numbasis = get_numbasis(x, x.kind_list[ikind])
        cn[x.kind_list[ikind]] = Matrix{T}(undef, numbasis, 0)
    end
    cn[atomkind_i] = hcat(cn[atomkind_i], cni)

    for ith_atom = 2:xsf.numatoms
        atomkind_i_symbol = Symbol(xsf.kinds[ith_atom])
        kanlayer = x.kanlayers[atomkind_i_symbol]
        basis = kanlayer.basis
        R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(xsf, ith_atom, basis.Rc)
        cni = kanlayer(R_js, atomkinds_j, x.kind_list, kanlayer.colors)
        cn[atomkind_i] = hcat(cn[atomkind_i], cni)
    end
    return Tuple(values(cn))
end
=#


function (x::KANbasis{numkinds})(xsfs::AbstractVector{XSFdata}) where {numkinds}
    numfiles = length(xsfs)
    ifile = 1
    xsf_i = xsfs[ifile]
    cn = x(xsf_i)
    T = eltype(cn[1])

    cnset = Vector{Matrix{T}}(undef, numkinds)
    for ikind = 1:numkinds
        numbasis = get_numbasis(x, x.kind_list[ikind])
        cnset[ikind] = Matrix{T}(undef, numbasis, 0)
    end
    for ikind = 1:numkinds
        cn_i = cn[ikind]
        cnset[ikind] = hcat(cnset[ikind], cn_i)
    end

    for ifile = 2:numfiles
        xsf_i = xsfs[ifile]
        cn = x(xsf_i)
        for ikind = 1:numkinds
            cn_i = cn[ikind]
            cnset[ikind] = hcat(cnset[ikind], cn_i)
        end
    end
    return cnset
end
