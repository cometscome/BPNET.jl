struct KAN_composit_layer{T} <: KAN_layer{T}
    layers::Vector{KAN_layer{T}}

    function KAN_composit_layer(xs::AbstractVector{KAN_layer{T}}) where {T}
        return new{T}(xs)
    end

    function KAN_composit_layer(xs::KAN_layer{T}) where {T}
        return new{T}([xs])
    end
end
export KAN_composit_layer



function Base.push!(x::KAN_composit_layer, xi::KAN_layer)
    push!(x.layers, xi)
end

function Base.display(x::KAN_composit_layer)
    println("---------------------------")
    println("num. of layers: $(length(x.layers))")
    for xi in x.layers
        println("---------------------------")
        display(xi)
        println("---------------------------")
    end
end

function (x::KAN_composit_layer{T})(xsfs::Vector{XSFdata}) where {T}
    numlayer = length(x.layers)
    i = 1
    xi = x.layers[i]
    cnset = xi(xsfs)

    for i = 2:numlayer
        xi = x.layers[i]
        cnset_i = xi(xsfs)
        for ikind = 1:xi.numkinds
            cn_i = cnset_i[ikind]
            cnset[ikind] = vcat(cnset[ikind], cn_i)
        end
    end
    return cnset
end

function get_numbasis(x::KAN_composit_layer)
    numbasis = 0
    for xi in x.layers
        numbasis += get_numbasis(xi)
    end
    return numbasis
end

