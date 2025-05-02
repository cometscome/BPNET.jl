using InteractiveUtils

mutable struct ChebyshevLayer{T1,T2} <: AbstractbasisLayer
    nc_R::Int64
    nc_θ::Int64
    radialbasis::T1
    angularbasis::T2
    Rc::Float64
    const Rmin::Float64
    const radialonly::Bool
end
export ChebyshevLayer

function Base.display(x::ChebyshevLayer)
    Rmin = x.Rmin
    Rc = x.Rc

    radialonly = x.radialonly
    println("Descriptor $(typeof(x))")
    println("Rmin = $(Rmin)")
    println("Rc = $(Rc)")

    if x.radialonly
        println("radialonly = $(radialonly)")
        display(x.radialbasis)
    else
        println("radial part:")
        display(x.radialbasis)
        println("angular part:")
        display(x.angularbasis)
    end


end


function get_numparams(m::ChebyshevLayer, numkinds)
    if m.radialonly
        nn = m.nc_R + 1
    else
        nn = m.nc_R + 1 + get_numparams(m.angularbasis)
    end
    #nn = m.nc_R + 1
    return ifelse(numkinds == 1, nn, 2nn)
end

function ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8, Rc_R=nothing, Rc_θ=nothing, radialonly=false)
    if isnothing(Rc_R)
        Rc_R = Rc
    end
    if isnothing(Rc_θ)
        Rc_θ = Rc
    end

    #nn = nc_R + nc_θ + 2
    if nc_θ == 0 && useYlm == false
        radialonly_in = true
    else
        radialonly_in = radialonly
    end

    radialbasis = KACbasis_radial(; polynomial_order=nc_R, Rmin, Rc=Rc_R)
    if radialonly
        angularbasis = nothing
    else
        angularbasis = KACbasis_angular(; polynomial_order=nc_θ, Rmin, Rc=Rc_θ)

    end
    Rc = max(Rc_R, Rc_θ)

    return ChebyshevLayer{typeof(radialbasis),typeof(angularbasis)}(nc_R, nc_θ, radialbasis, angularbasis, Rc, Rmin, radialonly_in)
end




export ChebyshevLayer
Flux.@layer ChebyshevLayer trainable = ()


function (m::ChebyshevLayer)(R_js::Vector{Vector{T}}) where {T<:Real}
    #radial
    y_R = m.radialbasis(R_js)
    #display(y_R)
    if m.radialonly
        y = y_R
    else
        y_θ = m.angularbasis(R_js)
        #display(y_θ)
        y = cat(y_R, y_θ, dims=1)
    end

    #y = y_R
    return y
end

function (m::ChebyshevLayer)(R_js::Vector{Vector{T}}, color_j::Matrix{<:Number}) where {T<:Real}
    y_R = m.radialbasis(R_js, color_j)
    #display(y_R)
    if m.radialonly
        y = y_R
    else
        y_θ = m.angularbasis(R_js, color_j)
        #display(y_θ)
        y = cat(y_R, y_θ, dims=1)
    end
    return y
end

function (m::ChebyshevLayer)(R_js::Matrix{T}, color_j::Matrix{<:Number}, numneighbors) where {T<:Real}
    y_R = m.radialbasis(R_js, color_j, numneighbors)
    #@code_warntype m.radialbasis(R_js, color_j, numneighbors)
    #display(y_R)
    if m.radialonly
        y = y_R
    else
        y_θ = m.angularbasis(R_js, color_j, numneighbors)
        #@code_warntype m.angularbasis(R_js, color_j, numneighbors)
        #display(y_θ)
        y = cat(y_R, y_θ, dims=1)
    end

    #error("c")
    return y
end


function (m::ChebyshevLayer)(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors) where {T<:Real}
    color_j = reshape(make_color(kinds_j::Vector{String}, kind_list, colors), :, 1)
    #radial
    y = m(R_js, color_j)
    return y
end

function (m::ChebyshevLayer)(R_js::Matrix{T}, kinds_j::Vector{String}, kind_list, colors, numatom) where {T<:Real}
    color_j = reshape(make_color(kinds_j::Vector{String}, kind_list, colors), :, 1)
    #radial
    y = m(R_js, color_j, numatom)
    return y
end

function get_Rc_radial(m::ChebyshevLayer)
    return m.radialbasis.Rc
end
function get_Rc_angular(m::ChebyshevLayer)
    if m.radialonly
        return 0
    else
        return m.angularbasis.Rc
    end
end
function get_Rmin_radial(m::ChebyshevLayer)
    return m.radialbasis.Rmin
end
function get_Rmin_angular(m::ChebyshevLayer)
    if m.radialonly
        return 0.0
    else
        return m.angularbasis.Rmin
    end
end

using FileIO

function write_descriptor(xsf_i::XSFdata, m::ChebyshevLayer, filename,
    kind_list; Rmax=8.0)
    numatoms = xsf_i.numatoms

    numkinds = length(kind_list)
    colors = color_vector(numkinds)

    data = Dict()
    data["numatoms"] = numatoms
    data["nc_R"] = m.nc_R

    data["nc_θ"] = m.nc_θ



    data["Rc_angular"] = get_Rc_angular(m)
    data["Rmin_angular"] = get_Rmin_angular(m)

    data["Rc_radial"] = get_Rc_radial(m)

    data["Rmin_radial"] = get_Rmin_radial(m)

    data["energy"] = get_energy(xsf_i)
    data["kind_list"] = kind_list
    data["colors"] = colors
    numkind = length(kind_list)

    for ith_atom = 1:numatoms
        R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(xsf_i, ith_atom, Rmax)
        if numkind == 1
            cn_i = m(R_js)
        else
            cn_i = m(R_js, atomkinds_j, kind_list, colors)
        end
        data["atomkind_$(ith_atom)"] = atomkind_i
        data["cn_$(ith_atom)_$(atomkind_i)"] = cn_i
        data["R_$(ith_atom)"] = R_i
        data["index_$(ith_atom)"] = index_i
        data["indices_j_$(ith_atom)"] = indices_j
    end
    save(filename * ".jld2", data)
end
export write_descriptor

function construct_KANfunction(m::ChebyshevLayer, scale_r, shift_r, scale_t, shift_t,
    Wkn_r, Wkn_r_s, Wkn_t, Wkn_t_s; dR=0.01, npoints=100, dθ=0.00)
    radialbasis = m.radialbasis
    ck_r, ck_r_s, bk_r, bk_r_s = construct_KANfunction(radialbasis, scale_r, shift_r, Wkn_r, Wkn_r_s; dR, npoints)
    if m.radialonly
        return ck_r, ck_r_s, bk_r, bk_r_s#, nothing, nothing, nothing, nothing
    else
        angularbasis = m.angularbasis
        ck_t, ck_t_s, bk_t, bk_t_s = construct_KANfunction(angularbasis, scale_t, shift_t, Wkn_t, Wkn_t_s; npoints, dθ)
        return ck_r, ck_r_s, bk_r, bk_r_s, ck_t, ck_t_s, bk_t, bk_t_s
    end
end

function construct_KANfunction(m::ChebyshevLayer, scale_r, shift_r, scale_t, shift_t,
    Wkn_r, Wkn_t; dR=0.01, npoints=100, dθ=0.00)
    radialbasis = m.radialbasis
    ck_r, bk_r = construct_KANfunction(radialbasis, scale_r, shift_r, Wkn_r; dR, npoints)
    if m.radialonly
        return ck_r, bk_r
    else
        angularbasis = m.angularbasis
        ck_t, bk_t = construct_KANfunction(angularbasis, scale_t, shift_t, Wkn_t; npoints, dθ)
        return ck_r, bk_r, ck_t, bk_t
    end
end


#=
function write_descriptor(m::ChebyshevLayer, filename, R_js::Vector{Vector{T}}) where {T<:Real}
    data = Dict()
    data["nc_R"] = m.nc_R
    data["nc_θ"] = m.nc_θ
    y = m(R_js)
    data["descriptor"] = y
    save(filename * ".jld2", data)
end
export write_descriptor
=#