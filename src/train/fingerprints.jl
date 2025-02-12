const num_of_maxparameters = 4


struct FingerPrint
    itype::Int64
    description::String
    atomtype::String
    nenv::Int64
    envtypes::Vector{String}
    rc_min::Float64
    rc_max::Float64
    sftype::String
    nsf::Int64
    nsfparam::Int64
    sf::Vector{Int64}
    sfparam::Matrix{Float64}
    sfenv::Matrix{Int64}
    neval::Int64
    sfval_min::Vector{Float64}
    sfval_max::Vector{Float64}
    sfval_avg::Vector{Float64}
    sfval_cov::Vector{Float64}

end



struct FingerPrint_params
    basistype::String
    num_kinds::Int64
    numparams::Int64
    params::Vector{Float64}
    startindex::Int64
    endindex::Int64
end



function get_singlefingerprints_info(fingerprint::FingerPrint, inputdim)
    fingerprint_parameters_set = Vector{FingerPrint_params}(undef, 1)

    fingerprint_parameters = fingerprint.sfparam[:, 1]
    num_kinds = 1
    startindex = 1
    endindex = length(fingerprint.sfparam[:, 1])
    numparams = inputdim
    fingerprint_parameters_set[1] = FingerPrint_params("any single basis", num_kinds, numparams, fingerprint_parameters, startindex, endindex)
    return fingerprint_parameters_set
end



function get_multifingerprints_info(fingerprint::FingerPrint)
    @assert fingerprint.sfparam[1, 1] == 0.0 "This finger print is not a multi version"
    fingerprint_parameters = fingerprint.sfparam[:, 1]
    num_kinds = Int(fingerprint_parameters[2])
    #display(fingerprint.sfparam[:, :])
    fingerprint_parameters_set = Vector{FingerPrint_params}(undef, num_kinds)

    #println(num_kinds)
    startindex = 1
    for ikind = 1:num_kinds
        istart = 3 + (ikind - 1) * (num_of_maxparameters + 2)
        ibasis = Int(fingerprint_parameters[istart])
        if ibasis == 1
            basistype = "Chebyshev"
        elseif ibasis == 2
            basistype = "Spline"
        elseif ibasis == 3
            basistype = "LJ"
        end

        numparams_i = Int(fingerprint_parameters[istart+1])
        params = fingerprint_parameters[istart+2:istart+1+num_of_maxparameters]
        #println(numparams_i)
        #println(params)
        endindex = startindex + numparams_i - 1

        fingerprint_parameters_set[ikind] = FingerPrint_params(basistype, num_kinds, numparams_i, params, startindex, endindex)
        startindex += numparams_i
    end
    return fingerprint_parameters_set
end

