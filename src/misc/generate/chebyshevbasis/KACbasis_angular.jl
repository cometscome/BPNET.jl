struct KACbasis_angular
    Rmin::Float64
    Rc::Float64
    polynomial_order::Int64
    #cn::Vector{Float64}
    #Tn::Vector{Float64}
end

function KACbasis_angular(; polynomial_order=10, Rmin=0.75, Rc=8)
    #cn = zeros(polynomial_order + 1 + polynomial_order + 1)
    #Tn = zeros(polynomial_order + 1)
    return KACbasis_angular(Rmin, Rc, polynomial_order)#, cn, Tn)
end

function Base.display(x::KACbasis_angular)
    Rmin = x.Rmin
    Rc = x.Rc
    polynomial_order = x.polynomial_order
    println("KACbasis_angular $(typeof(x))")
    println("Rmin = $(Rmin)")
    println("Rc = $(Rc)")
    println("polynomial_order = $(polynomial_order)")
end

function get_numparams(m::KACbasis_angular)
    return m.polynomial_order + 1
end

export KACbasis_angular
Flux.@layer KACbasis_angular trainable = ()

function (m::KACbasis_angular)(x::Matrix{<:Number})
    y = KACbasis_angular_forward(x, m.Rc,
        m.polynomial_order)
    return y
end

function (m::KACbasis_angular)(R_js::Vector{Vector{T}}) where {T<:Real}
    numatom = length(R_js)
    cn = zeros(m.polynomial_order + 1)
    for j = 1:numatom
        cn += KACbasis_angular_forward_j(R_js, j, numatom,
            m.Rc, m.Rmin,
            m.polynomial_order)
    end
    cn = reshape(cn, :, 1)
    return cn
end

function (m::KACbasis_angular)(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors) where {T<:Real}
    color_j = reshape(make_color(kinds_j::Vector{String}, kind_list, colors), :, 1)
    y = m(R_js, color_j)
    return y
end

function (m::KACbasis_angular)(R_js::Vector{Vector{T}}, color_j::Matrix{<:Number}) where {T<:Real}
    numatom = length(R_js)
    cn = zeros(m.polynomial_order + 1 + m.polynomial_order + 1)
    for j = 1:numatom
        cn += KACbasis_angular_forward_j(R_js, j, numatom,
            m.Rc, m.Rmin,
            m.polynomial_order, color_j)
    end
    cn = reshape(cn, :, 1)
    return cn
end

function (m::KACbasis_angular)(R_js::Matrix{T}, color_j::Matrix{<:Number}, numneighbors) where {T<:Real}
    #numatom = length(R_js)
    cn = zeros(m.polynomial_order + 1 + m.polynomial_order + 1)
    Tn = zeros(m.polynomial_order + 1)
    #cn = m.cn
    #Tn = m.Tn

    #cn .= 0
    #Tn .= 0

    for j = 1:numneighbors
        KACbasis_angular_forward_j_color!(cn, R_js, j, numneighbors,
            m.Rc, m.Rmin,
            m.polynomial_order, color_j, Tn)
        #println(typeof(cj))
        #cn += cj
        #cn += KACbasis_angular_forward_j(R_js, j, numneighbors,
        #    m.Rc, m.Rmin,
        #    m.polynomial_order, color_j)
    end
    cn = reshape(cn, :, 1)
    return cn
end


function KACbasis_angular_forward_j(R_js::Vector{Vector{T}}, j, numneighbors,
    Rc, Rmin, polynomial_order) where {T<:Real}
    numatom = numneighbors

    Rij = R_js[j]
    dij = norm(Rij)
    nc = polynomial_order
    cn = zeros(nc + 1)
    nn = 0:nc
    window_ij = cutoff_x(dij, Rc, Rmin)
    for k = j+1:numatom
        Rik = R_js[k]
        dik = norm(Rik)
        if dik > Rc && dik < Rmin
        else
            window_ik = cutoff_x(dik, Rc, Rmin)
            dijdik = dij * dik
            invdijdik = 1 / dijdik
            cos_ijk = dot(Rik, Rij) * invdijdik
            #println(cos_ijk, "\t ", dot(Rik, Rij), "\t", invdijdik)
            #display(Rik)
            #display(Rij)
            r = cos_ijk
            r0 = -1
            r1 = 1
            x = (2 * r - r0 - r1) / (r1 - r0)
            x = ifelse(x > 1, 1.0, x)
            x = ifelse(x < -1, -1.0, x)
            Tn = cos.(nn * acos(x))
            #display(Tn)
            cn += window_ij * window_ik * Tn
        end
    end

    return cn
end

function KACbasis_angular_forward_j(R_js::Vector{Vector{T}}, j, numatom,
    Rc, Rmin, polynomial_order, color_j) where {T<:Real}

    Rij = R_js[j]
    dij = norm(Rij)
    nc = polynomial_order
    cn = zeros(nc + 1 + nc + 1)
    #nn = collect(0:nc)
    Tn = zeros(nc + 1)
    window_ij = cutoff_x(dij, Rc, Rmin)
    s_j = color_j[j]
    for k = j+1:numatom

        Rik = R_js[k]
        dik = norm(Rik)
        if dik > Rc && dik < Rmin
        else
            s_k = color_j[k]
            window_ik = cutoff_x(dik, Rc, Rmin)
            dijdik = dij * dik
            invdijdik = 1 / dijdik
            cos_ijk = dot(Rik, Rij) * invdijdik
            #println(cos_ijk, "\t ", dot(Rik, Rij), "\t", invdijdik)
            #display(Rik)
            #display(Rij)
            r = cos_ijk
            r0 = -1
            r1 = 1
            x = (2 * r - r0 - r1) / (r1 - r0)
            x = ifelse(x > 1, 1.0, x)
            x = ifelse(x < -1, -1.0, x)

            chebyshev_vec!(Tn, x, nc)
            #Tn = cos.(nn * acos(x))
            for nn = 1:nc+1
                cn[nn] += window_ij * window_ik * Tn[nn]
                cn[nc+nn+1] += window_ij * window_ik * Tn[nn] * s_j * s_k
            end
            #cn[1:nc+1] .+= window_ij * window_ik * Tn #wwTn
            #cn[nc+2:end] .+= Tn .* (s_j * s_k * window_ij * window_ik) #wwTn_scaled
            #cn += cat(wwTn, wwTn * s_j * s_k, dims=1)
            error("no!!!")

            #cn += window_ij * window_ik * Tn
        end
    end

    return cn
end

function KACbasis_angular_kth!(cn, nc, k, R_js, Rij, Rik, dij, window_ij, s_j, Tn, Rc, Rmin, color_j)
    for i = 1:3
        Rik[i] = R_js[i, k]
    end

    dik = norm(Rik)
    if dik > Rc && dik < Rmin
    else
        s_k = color_j[k]
        window_ik = cutoff_x(dik, Rc, Rmin)
        dijdik = dij * dik
        invdijdik = 1 / dijdik
        cos_ijk = dot(Rik, Rij) * invdijdik

        r = cos_ijk
        r0 = -1
        r1 = 1
        x = (2 * r - r0 - r1) / (r1 - r0)
        x = ifelse(x > 1, 1.0, x)
        x = ifelse(x < -1, -1.0, x)

        factor1 = window_ij * window_ik
        factor2 = window_ij * window_ik * s_j * s_k
        chebyshev_vec_and_cc!(Tn, x, nc, cn, factor1, factor2)
    end
end


function KACbasis_angular_forward_j_color!(cn, R_js::Matrix{T}, j, numatom,
    Rc, Rmin, polynomial_order, color_j, Tn) where {T<:Real}


    Rij = zeros(3)
    Rik = zeros(3)
    for k = 1:3
        Rij[k] = R_js[k, j]
    end
    #Rij = R_js[j]
    dij = norm(Rij)
    if dij > Rc && dij < Rmin
        return
    end

    nc = polynomial_order
    #cn = zeros(nc + 1 + nc + 1) * sum(R_js)
    #val = sum(R_js)
    #cn .+= val
    #cn = fill(val, nc + 1 + nc + 1)
    #cn = fill(sum(R_js), nc + 1 + nc + 1)

    #return


    #nn = collect(0:nc)
    #Tn = zeros(nc + 1)
    window_ij = cutoff_x(dij, Rc, Rmin)
    s_j = color_j[j]
    for k = j+1:numatom
        KACbasis_angular_kth!(cn, nc, k, R_js, Rij, Rik, dij, window_ij, s_j, Tn, Rc, Rmin, color_j)


        #=
        #Rik = R_js[k]
        for i = 1:3
            Rik[i] = R_js[i, k]
        end
        dik = norm(Rik)
        if dik > Rc && dik < Rmin
        else
            s_k = color_j[k]
            window_ik = cutoff_x(dik, Rc, Rmin)
            dijdik = dij * dik
            invdijdik = 1 / dijdik
            cos_ijk = dot(Rik, Rij) * invdijdik
            #println(cos_ijk, "\t ", dot(Rik, Rij), "\t", invdijdik)
            #display(Rik)
            #display(Rij)
            r = cos_ijk
            r0 = -1
            r1 = 1
            x = (2 * r - r0 - r1) / (r1 - r0)
            x = ifelse(x > 1, 1.0, x)
            x = ifelse(x < -1, -1.0, x)

            factor = window_ij * window_ik * s_j * s_k
            chebyshev_vec_and_cc!(Tn, x, nc, cn, factor)

            #chebyshev_vec!(Tn, x, nc)
            #for nn = 1:nc+1
            #    cn[nn] += window_ij * window_ik * Tn[nn]
            #    cn[nc+nn+1] += window_ij * window_ik * Tn[nn] * s_j * s_k
            #end
            #Tn = cos.(nn * acos(x))
            #wwTn = window_ij * window_ik * Tn
            #cn += cat(wwTn, wwTn * s_j * s_k, dims=1)
            #cn[1:nc+1] .+= window_ij * window_ik * Tn #wwTn
            #cn[nc+2:end] .+= Tn .* (s_j * s_k * window_ij * window_ik) #wwTn_scaled

            #cn += window_ij * window_ik * Tn
        end
        =#
    end

    return #cn
end
@inline function chebyshev_vec!(out, x, n)
    #out = Vector{Float64}(undef, n + 1)
    out[1] = 1.0
    n == 0 && return out
    out[2] = x
    for k = 2:n
        out[k+1] = 2x * out[k] - out[k-1]
    end
    #return out
end

@inbounds function chebyshev_vec_and_cc!(out, x, n, cc, factor1, factor2)
    #out = Vector{Float64}(undef, n + 1)
    out[1] = 1.0
    cc[1] += out[1] * factor1
    cc[1+n+1] += out[1] * factor2
    n == 0 && return out
    out[2] = x
    cc[2] += out[2] * factor1
    cc[2+n+1] += out[2] * factor2
    for k = 2:n
        out[k+1] = 2x * out[k] - out[k-1]
        cc[k+1] += out[k+1] * factor1
        cc[k+1+n+1] += out[k+1] * factor2
    end
    #return out
end

function get_coeffs(m::KACbasis_angular)
    return m.Wkn.weight
end
export get_coeffs



function construct_KANfunction(m::KACbasis_angular, scale, shift, Wkn, Wkn_s; npoints=100)
    coss = collect(range(-1.05, 1.05, length=npoints))
    x = reshape(coss, 1, :)
    polynomial_order = m.polynomial_order

    chebyshev_polys = compute_chebyshev_polynomials(x, polynomial_order)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)

    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    bn = -shift ./ scale

    ck = Wkn * chebyshev_basis
    ck_s = Wkn_s * chebyshev_basis

    bk = Wkn * bn[1:polynomial_order+1]
    bk_s = Wkn_s * bn[polynomial_order+1+1:end]
    return ck, ck_s, bk, bk_s
end

function construct_KANfunction(m::KACbasis_angular, scale, shift, Wkn; npoints=100, dθ=0.00)
    coss = collect(range(-1 - dθ, 1 + dθ, length=npoints))
    x = reshape(coss, 1, :)
    polynomial_order = m.polynomial_order

    #chebyshev_polys = compute_chebyshev_polynomials(x, polynomial_order)
    chebyshev_polys = compute_chebyshev_polynomials_custom(x, polynomial_order)
    chebyshev_polys = map(x -> sum(x, dims=1), chebyshev_polys)

    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    bn = -shift ./ scale

    ck = Wkn * (chebyshev_basis ./ scale)

    bk = Wkn * bn#[1:polynomial_order+1]

    #println("bn_angular: ")
    #display(bn)
    #display(bk)
    #display(ck)

    return ck, bk
end
export construct_KANfunction