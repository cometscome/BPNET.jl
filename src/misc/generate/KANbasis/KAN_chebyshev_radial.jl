

struct KAN_Chebyshev_radiallayer{T} <: KAN_Chebyshev_layer{T}
    an::Vector{T}
    bn::Vector{T}
    basis::KACbasis_radial
    kind_list::Vector{String}
    numbasis::Int64
    numkinds::Int64
    colors::Vector{T}
    #Rmin::T
    #Rc::T
    #polynomial_order::Int64

    function KAN_Chebyshev_radiallayer(an, bn, radialbasis, kind_list, numbasis, numkinds, colors)
        T = eltype(an)
        return new{T}(an, bn, radialbasis, kind_list, numbasis, numkinds, colors)
    end
end



function KAN_Chebyshev_radiallayer(kind_list; polynomial_order=10, Rmin=0.75, Rc=8.0)
    numkinds = length(kind_list)
    numbasis = polynomial_order + 1
    numbasis *= ifelse(numkinds > 1, 2, 1)

    an = ones(numbasis)
    bn = zeros(numbasis)
    radialbasis = KACbasis_radial(; polynomial_order, Rmin, Rc)

    colors = color_vector(numkinds)

    return KAN_Chebyshev_radiallayer(an, bn, radialbasis, kind_list, numbasis, numkinds, colors)
end

export KAN_Chebyshev_radiallayer
Flux.@layer KAN_Chebyshev_radiallayer trainable = (an, bn)

