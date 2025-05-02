using FluxKAN
using LinearAlgebra



mutable struct KACbasis_radial
    Rmin::Float64
    Rc::Float64
    polynomial_order::Int64
    #cn::Vector{Float64}
    #Tn::Vector{Float64}
end

function KACbasis_radial(; polynomial_order=10, Rmin=0.75, Rc=8)
    #cn = zeros(polynomial_order + 1 + polynomial_order + 1)
    #Tn = zeros(polynomial_order + 1)
    return KACbasis_radial(Rmin, Rc, polynomial_order)#, cn, Tn)
end

function Base.display(x::KACbasis_radial)
    Rmin = x.Rmin
    Rc = x.Rc
    polynomial_order = x.polynomial_order
    println("KACbasis_radial $(typeof(x))")
    println("Rmin = $(Rmin)")
    println("Rc = $(Rc)")
    println("polynomial_order = $(polynomial_order)")
end

export KACbasis_radial
Flux.@layer KACbasis_radial trainable = ()

function (m::KACbasis_radial)(x::Matrix{<:Number})
    y = KACbasis_radial_forward(x, m.Rc, m.polynomial_order, m.Rmin)
    return y
end

#function (m::KACbasis_radial)(x::Matrix{<:Number}, color_j::Matrix{<:Number})
#    y = KACbasis_radial_forward(x, color_j, m.Rc, m.polynomial_order, m.Rmin)
#    return y
#end

function (m::KACbasis_radial)(R_js::Vector{Vector{T}}) where {T<:Real}
    Rij = norm.(R_js)
    x = reshape(Rij, :, 1)
    y = m(x)
    return y
end

function (m::KACbasis_radial)(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors) where {T<:Real}
    color_j = reshape(make_color(kinds_j::Vector{String}, kind_list, colors), :, 1)
    y = m(R_js, color_j)
    return y
end

function (m::KACbasis_radial)(R_js::Vector{Vector{T}}, color_j::Matrix{<:Number}) where {T<:Real}
    Rij = norm.(R_js)
    x = reshape(Rij, :, 1)
    #display(x)
    #y = m(x, color_j)
    #display(y)
    y = KACbasis_radial_forward(x, color_j, m.Rc, m.polynomial_order, m.Rmin)
    return y
end



function (m::KACbasis_radial)(R_js::Matrix{T}, color_j::Matrix{<:Number}, numneighbors) where {T<:Real}
    #Rij = norm.(R_js)
    #cn = m.cn
    #Tn = m.Tn

    #cn .= 0
    #Tn .= 0

    cn = zeros(m.polynomial_order + 1 + m.polynomial_order + 1)
    Tn = zeros(m.polynomial_order + 1)
    for j = 1:numneighbors
        KACbasis_radial_forward_j_color!(cn, R_js, j,
            m.Rc, m.Rmin,
            m.polynomial_order, color_j, Tn)
    end
    return cn


    Rij = [norm(R_js[:, i]) for i in 1:numneighbors]
    x = reshape(Rij, :, 1)
    #y = m(x, color_j)
    #display(y)
    #display(x)
    y = KACbasis_radial_forward(x, color_j, m.Rc, m.polynomial_order, m.Rmin)

    display(cn)
    display(y)
    #@code_warntype KACbasis_radial_forward(x, color_j, m.Rc, m.polynomial_order, m.Rmin)
    error("r")
    return y
end

function KACbasis_radial_forward_j_color!(cn, R_js, j,
    Rc, Rmin,
    polynomial_order, color_j, Tn)

    Rij = zeros(3)
    for k = 1:3
        Rij[k] = R_js[k, j]
    end
    dij = norm(Rij)
    nc = polynomial_order

    if dij > Rc && dij < Rmin
        return
    end

    window_ij = cutoff_x(dij, Rc, Rmin)
    s_j = color_j[j]

    x = renormalize_x(dij, Rc)
    factor1 = window_ij
    factor2 = window_ij * s_j
    chebyshev_vec_and_cc!(Tn, x, nc, cn, factor1, factor2)

end



function color_vector(numkinds)
    colors = zeros(numkinds)
    if numkinds == 1
        return colors[1] = 1
    end
    if numkinds % 2 == 0 #even
        for i = 1:(numkinds÷2)
            colors[i] = i
            colors[i+numkinds÷2] = -i
        end
    else
        colors[(numkinds-1)÷2+1] = 0
        for i = 1:((numkinds-1)÷2)
            colors[i] = i
            colors[(numkinds-1)÷2+1+i] = -i
        end
    end
    return colors
end
export color_vector

function make_color(kinds_j::Vector{String}, kind_list, colors)
    color_j = zeros(length(kinds_j))
    #numkinds = length(kind_list)
    for j = 1:length(color_j)
        index_j = findfirst(x -> x == kinds_j[j], kind_list)
        @assert typeof(index_j) <: Integer "kind_list might be not correct. kind_list is $kind_list but kinds_j[j] is $(kinds_j[j])"
        color_j[j] = colors[index_j]
    end
    return color_j
end


function KACbasis_radial_forward(x, Rc, polynomial_order, Rmin)
    xtilde = renormalize_x.(x, Rc)
    fc = cutoff_x.(x, Rc, Rmin)
    chebyshev_polys = compute_chebyshev_polynomials(xtilde, polynomial_order)
    chebyshev_polys = map(y -> y .* fc, chebyshev_polys)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)
    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    return chebyshev_basis
end

function KACbasis_radial_forward(x, color_j, Rc, polynomial_order, Rmin)



    xtilde = renormalize_x.(x, Rc)
    fc = cutoff_x.(x, Rc, Rmin)
    #chebyshev_polys = compute_chebyshev_polynomials(xtilde, polynomial_order)
    chebyshev_polys = compute_chebyshev_polynomials_custom(xtilde, polynomial_order)

    chebyshev_polys = map(y -> y .* fc, chebyshev_polys)
    #chebyshev_polys_s = deepcopy(chebyshev_polys)
    chebyshev_polys_s = copy(chebyshev_polys)
    chebyshev_polys_s = map(y -> y .* color_j, chebyshev_polys_s)

    #display(chebyshev_polys_s)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)
    chebyshev_polys_s = map(x -> sum(x, dims=1), chebyshev_polys_s)
    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    chebyshev_basis_s = cat(chebyshev_polys_s..., dims=1)
    #println(typeof(chebyshev_polys_s))
    #println(typeof(chebyshev_polys))
    #display(chebyshev_basis)
    #display(chebyshev_basis_s)
    chebyshev_basis = cat(chebyshev_basis, chebyshev_basis_s, dims=1)
    #chebyshev_basis = cat(chebyshev_polys, chebyshev_basis_s)
    return chebyshev_basis
end

function construct_KANfunction(m::KACbasis_radial, scale, shift, Wkn, Wkn_s; dR=0.01, npoints=100)
    Rc = m.Rc
    Rmin = m.Rmin
    rs = collect(range(Rmin + dR, Rc, length=npoints))
    x = reshape(rs, 1, :)
    polynomial_order = m.polynomial_order

    xtilde = renormalize_x.(x, Rc)
    fc = cutoff_x.(x, Rc, Rmin)
    chebyshev_polys = compute_chebyshev_polynomials(xtilde, polynomial_order)
    chebyshev_polys = map(y -> y .* fc, chebyshev_polys)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)

    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    bn = -shift ./ scale

    ck = Wkn * (chebyshev_basis ./ scale)
    ck_s = Wkn_s * (chebyshev_basis ./ scale)

    bk = Wkn * bn[1:polynomial_order+1]
    bk_s = Wkn_s * bn[polynomial_order+1+1:end]
    return ck, ck_s, bk, bk_s
end

function construct_KANfunction(m::KACbasis_radial, scale, shift, Wkn; dR=0.01, npoints=100)
    Rc = m.Rc
    Rmin = m.Rmin
    rs = collect(range(Rmin + dR, Rc, length=npoints))
    x = reshape(rs, 1, :)
    polynomial_order = m.polynomial_order

    xtilde = renormalize_x.(x, Rc)
    fc = cutoff_x.(x, Rc, Rmin)
    chebyshev_polys = compute_chebyshev_polynomials(xtilde, polynomial_order)
    chebyshev_polys = map(y -> y .* fc, chebyshev_polys)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)

    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    bn = -shift ./ scale
    #println("bn_radial: ")
    #display(bn)

    ck = Wkn * (chebyshev_basis ./ scale)

    bk = Wkn * bn
    #display(bk)
    #display(ck)
    return ck, bk
end

export construct_KANfunction