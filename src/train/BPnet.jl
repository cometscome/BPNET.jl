using Flux
using FluxKAN

using Optimisers
struct BPnet{N,T} # Parameter to avoid type instability. T<:Chain or T<:BPnet{N,Chain}
    chains::NTuple{N,T}
end
export BPnet

#function (m::BPnet{N,T})(x) where {N,T<:Chain}
#    energies = sum(map(apply_bpmodel, m.chains, x))
#    return energies
#end

function BPnet(inputdata, fingerprintparams::Vector{Vector{FingerPrint_params}})
    numbasiskinds = inputdata["numbasiskinds"]
    atomtypes = inputdata["atomtypes"]
    keys = Symbol.(atomtypes)

    models = make_multimodel(atomtypes, fingerprintparams, inputdata, numbasiskinds)


    return models
end



function (m::BPnet{N,T})(x) where {N,T}

    energies = apply_bpmultimodel(m.chains[1], x[1])
    for i = 2:N
        energies += apply_bpmultimodel(m.chains[i], x[i])
    end
    #@code_warntype map(apply_bpmultimodel, m.chains, x)
    #error("dd")

    #energies = sum(map(apply_bpmultimodel, m.chains, x))


    return energies
end



function apply_bpmodel(model_i, x)
    #display(model_i)
    #println(typeof(x.data))
    #println(size(x.data))

    energies_i = model_i(x.data)#Lux.apply(model_i, x.data, ps, st)
    #println(typeof(energies_i))
    #println(size(energies_i))
    energies = energies_i * x.labels
    return energies
end

function apply_bpmultimodel(model_i, x)
    #display(model_i)
    #println(typeof(x.data))
    #println(size(x.data))
    #display(apply_model)
    energies_i = sum(map(apply_model, model_i, x.data))

    #energies_i = model_i(x.data)#Lux.apply(model_i, x.data, ps, st)
    #println(typeof(energies_i))
    #println(size(energies_i))
    energies = energies_i * x.labels
    return energies
end

function apply_model(model, x)
    return model(x)
end

function convertstring2function(activation)
    if activation == "tanh"
        return tanh
    elseif activation == ""
        return ""
    else
        error("$activation is not supported")
    end
end



function make_multimodel(atomtypes, fingerprintparams, inputdata, numbasiskinds)
    model_total = NTuple{numbasiskinds,Flux.Chain}[]#BPnet{numbasiskinds,Flux.Chain}[]
    kan = inputdata["kan"]
    resnet = inputdata["resnet"]

    for (itype, name) in enumerate(atomtypes)
        fingerprint_param = fingerprintparams[itype]

        models = Flux.Chain[]
        for ikind = 1:numbasiskinds
            if numbasiskinds == 1
                layers = inputdata[name]["layers"]
                activations_ikind = convertstring2function.(inputdata[name]["activations"])
                if kan
                    kantype = inputdata[name]["kantype"]
                    orders = inputdata[name]["orders"]
                end
            else
                layers = inputdata[name]["layers"][ikind]
                activations_ikind = convertstring2function.(inputdata[name]["activations"][ikind])
                if kan
                    kantype = inputdata[name]["kantype"][ikind]
                    orders = inputdata[name]["orders"][ikind]
                end
            end

            numparams = fingerprint_param[ikind].numparams
            #println(layers)
            #println(activations_ikind)

            if length(layers) > 0
                layerstructure = zeros(Int64, length(layers) + 2)
                layerstructure[1] = numparams
                layerstructure[2:end-1] = layers
                layerstructure[end] = 1

                activations = Vector{Any}(undef, length(layers) + 1)
                activations[1:end-1] = activations_ikind
                activations[end] = nothing


                if kan
                    model = make_kanmodel(layerstructure, activations, kantype, orders)
                else
                    model = make_densemodel(layerstructure, activations, resnet)
                end
            else
                layerstructure = zeros(Int64, 2)
                layerstructure[1] = numparams
                layerstructure[2] = 1
                activations = Vector{Any}(undef, 1)
                activations[end] = nothing
                model = Chain(Dense(numparams, 1))
            end

            #display(model)
            push!(models, model)
        end
        model_itype = Tuple(models)
        display(model_itype)
        push!(model_total, model_itype)
    end
    models = BPnet{length(model_total),eltype(model_total)}(Tuple(model_total))
    display(models)
    return models
end

function BPnet(layerstructures, layersactivations, keys, inputdims)

    models = Chain[]
    for name in keys
        #for itype = 1:ntypes
        layer = getfield(layerstructures, name)
        layerstructure = zeros(Int64, length(layer) + 2)

        layerstructure[1] = getfield(inputdims, name)#get_inputdim(dataset, name)
        layerstructure[2:end-1] = layer
        layerstructure[end] = 1

        activations = Vector{Any}(undef, length(layer) + 1)
        activations[1:end-1] = getfield(layersactivations, name)
        activations[end] = nothing

        model = make_densemodel(layerstructure, activations)
        push!(models, model)
    end
    return BPnet{length(models),eltype(models)}(Tuple(models))
end


function BPnet(dataset::BP_dataset{keys}, layerstructures, layersactivations) where {keys}
    inputdimarray = []
    for name in keys
        push!(inputdimarray, get_inputdim(dataset, name))
    end
    inputdims = NamedTuple{keys}(inputdimarray)
    return BPnet(layerstructures, layersactivations, keys, inputdims)
end




function make_densemodel(layers, activations, resnet=false)
    #layers = [2,10,10,3]
    numlayers = length(layers) - 1
    modellist = []
    for i = 1:numlayers
        istart = layers[i]
        iend = layers[i+1]
        if activations[i] == nothing
            d = Flux.Dense(istart, iend)
        else
            if istart == iend
                if resnet
                    d = Flux.Parallel(+, x -> x, Dense(istart, iend, activations[i]))
                else
                    d = Flux.Dense(istart, iend, activations[i])
                end
            else
                d = Flux.Dense(istart, iend, activations[i])
            end
        end

        push!(modellist, d)
    end
    return Flux.Chain(modellist...)
end

function make_kanmodel(layers, activations, kantype, orders)
    #layers = [2,10,10,3]
    numlayers = length(layers) - 1
    modellist = []
    i = 1
    istart = layers[i]
    iend = layers[i+1]
    if activations[i] == nothing
        d = Dense(istart, iend)
    else
        d = Dense(istart, iend, activations[i])
    end
    push!(modellist, d)
    for i = 2:numlayers
        istart = layers[i]
        iend = layers[i+1]
        if kantype[i-1] == "KAC"
            d = KACnet(istart, iend, polynomial_order=orders[i-1])
        elseif kantype[i-1] == "KAL"
            d = KALnet(istart, iend, polynomial_order=orders[i-1])
        else
            error("$(kantype[i-1]) is not supported in KAN")
        end
        push!(modellist, d)
    end

    return Chain(modellist...)
end

