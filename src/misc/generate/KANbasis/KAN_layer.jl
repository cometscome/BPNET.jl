
import Enzyme.EnzymeRules: forward, reverse, augmented_primal
using Enzyme.EnzymeRules
import Enzyme

struct NormalizeLayer{TA,TB}
    an::TA
    bn::TB
end
function (m::NormalizeLayer)(x)
    y = zero(x)
    for i = 1:length(x)
        y[i] = (x[i] - m.bn[i]) / m.an[i]
    end
    return y
end
#(m::NormalizeLayer)(x) = (x .- m.bn) ./ m.an
Flux.@layer NormalizeLayer trainable = (an, bn)
export NormalizeLayer


struct KAN_Layer{TB<:AbstractbasisLayer,TC,T}
    #an::T
    #bn::T
    normalizelayer::T
    basis::TB
    kind_list::Vector{String}
    numbasis::Int64
    numkinds::Int64
    colors::TC

    function KAN_Layer(an, bn, basis::AbstractbasisLayer, kind_list, numbasis, numkinds, colors)
        TB = typeof(basis)
        #T = typeof(an)
        TC = typeof(colors)
        normalizelayer = NormalizeLayer(an, bn)
        #return new{TB,T,TC}(an, bn, basis, kind_list, numbasis, numkinds, colors)
        return new{TB,TC,typeof(normalizelayer)}(normalizelayer, basis, kind_list, numbasis, numkinds, colors)
    end

    function KAN_Layer(an, bn, basis, kind_list)
        numkinds = length(kind_list)
        numbasis = get_numparams(basis, numkinds)
        colors = color_vector(numkinds)
        normalizelayer = NormalizeLayer(an, bn)
        #T = typeof(an)
        TC = typeof(colors)
        return new{typeof(basis),TC,typeof(normalizelayer)}(normalizelayer, basis, kind_list, numbasis, numkinds, colors)
        #return new{typeof(basis),T,TC}(an, bn, basis, kind_list, numbasis, numkinds, colors)
    end
    function KAN_Layer(normalizelayer, basis, kind_list, numbasis, numkinds, colors)
        TB = typeof(basis)
        #T = typeof(an)
        TC = typeof(colors)
        return new{TB,TC,typeof(normalizelayer)}(normalizelayer, basis, kind_list, numbasis, numkinds, colors)
    end
end
export KAN_Layer

include("KAN_basis.jl")

export KAN_Layer
Flux.@layer KAN_Layer trainable = (normalizelayer,)

function get_numbasis(m::KAN_Layer)
    return m.numbasis
end
export get_numbasis

function KAN_Layer(basis, kind_list)
    numkinds = length(kind_list)
    numbasis = get_numparams(basis, numkinds)
    an = ones(numbasis)
    bn = zeros(numbasis)

    return KAN_Layer(an, bn, basis, kind_list)
end

function (x::KAN_Layer)(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors) where {T<:Real}
    y = kanlayer_forward(R_js, kinds_j, kind_list, colors, x.basis, x.normalizelayer.an, x.normalizelayer.bn)
    return y
end

function (x::KAN_Layer)(R_js::Matrix{T}, kinds_j::Vector{String}, kind_list, colors, numneighbors) where {T<:Real}
    y = kanlayer_forward_matrix(R_js, kinds_j, kind_list, colors, x.basis, x.normalizelayer, numneighbors)
    #@code_warntype kanlayer_forward_matrix(R_js, kinds_j, kind_list, colors, x.basis, x.normalizelayer, numneighbors)
    #error("dd")
    return y
end

function kanlayer_forward(R_js::Vector{Vector{T}}, kinds_j::Vector{String}, kind_list, colors, basis::TB, an, bn) where {T<:Real,TB<:AbstractbasisLayer}
    y = basis(R_js, kinds_j, kind_list, colors)
    output = (y .- bn) ./ an
    return output
end

function kanlayer_forward_matrix(R_js::Matrix{T}, kinds_j::Vector{String}, kind_list, colors, basis::TB, normalizelayer, numneighbors) where {T<:Real,TB<:AbstractbasisLayer}
    y = basis(R_js, kinds_j, kind_list, colors, numneighbors)
    #@code_warntype basis(R_js, kinds_j, kind_list, colors, numneighbors)
    #error("m")
    #return sum(y, dims=1)
    output = normalizelayer(y)
    return output#sum(y, dims=1)
end
export kanlayer_forward

#=



function augmented_primal(
    ::RevConfigWidth{1},
    f::Enzyme.Const{typeof(kanlayer_forward)},
    ::Type{Enzyme.Active},
    R_js::Enzyme.Duplicated{Vector{Vector{Float64}}},
    kinds_j::Enzyme.Const{Vector{String}},
    kind_list::Enzyme.Const,
    colors::Enzyme.Const,
    basis::Enzyme.Const,
    an::Enzyme.Const{Vector{Float64}},
    bn::Enzyme.Const{Vector{Float64}}
)
    y = basis.val(R_js.val, kinds_j.val, kind_list.val, colors.val)
    output = (y .- bn.val) ./ an.val

    # y を保存すれば reverse で使える（必要なら basis output も保存）
    return Enzyme.AugmentedReturn(output, nothing, (deepcopy(y),))
end


function reverse(
    ::RevConfigWidth{1},
    f::Enzyme.Const{typeof(kanlayer_forward)},
    dret::Enzyme.Active,
    tape,
    R_js::Enzyme.Duplicated{Vector{Vector{Float64}}},
    kinds_j::Enzyme.Const{Vector{String}},
    kind_list::Enzyme.Const,
    colors::Enzyme.Const,
    basis::Enzyme.Const,
    an::Enzyme.Const{Vector{Float64}},
    bn::Enzyme.Const{Vector{Float64}}
)
    y = tape[1]

    # doutput/dy = 1/an, chain rule: dL/dy = dL/doutput * doutput/dy
    doutput_dy = 1.0 ./ an.val
    dL_dy = dret.val .* doutput_dy

    # call Enzyme again: define a closure that computes basis output
    basis_output = (R_js_internal) -> basis.val(R_js_internal, kinds_j.val, kind_list.val, colors.val)

    # autodiff this closure with respect to R_js
    Enzyme.autodiff(
        Reverse,
        Const(basis_output),
        Active,
        Duplicated(R_js.val, R_js.dval)
    )

    # scale result manually by dL_dy
    for i in eachindex(R_js.dval)
        for k = 1:3
            R_js.dval[i][k] *= dL_dy[i]  # ← this assumes dL_dy has same length as R_js
        end
    end

    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

=#