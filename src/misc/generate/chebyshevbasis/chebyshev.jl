function renormalize_x(x, Rc)
    return 2x / Rc - 1
end

function cutoff_x(x, Rc, Rmin)
    if Rmin < x < Rc
        return (cos(pi * x / Rc) + 1) / 2
    else
        return zero(x)
    end
end

function compute_chebyshev_polynomials_custom(x, order)
    # Base case polynomials P0 and P1
    P0 = zero(x)
    T = typeof(P0)
    fill!(P0, 1)
    #P0 = ones(eltype(x), size(x)...)#x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0
        #return P0
        return T[P0]
    end
    P1 = copy(x)
    chebyshev_polys = T[P0, P1]

    # Compute higher order polynomials using recurrence
    for n = 1:order-1
        Cp1 = chebyshev_polys[end]
        Cp0 = chebyshev_polys[end-1]

        Pn = map((x, cp1, cp0) -> 2 * x * cp1 - cp0, x, Cp1, Cp0)
        #Pn = 2 .* x .* chebyshev_polys[end] .- chebyshev_polys[end-1] #2x Tn - T_{n-1}
        #Pn = ((2.0 * n + 1.0) .* x .* chebyshev_polys[end] - n .* chebyshev_polys[end-1]) ./ (n + 1.0)
        push!(chebyshev_polys, Pn)
    end
    return chebyshev_polys
end
export compute_chebyshev_polynomials_custom