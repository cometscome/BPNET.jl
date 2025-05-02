using Flux
struct BPChain_Flux{L}
    layers::L
end
export BPChain_Flux

function BPChain_Flux(inputdata, fingerprintparams::Vector{Vector{FingerPrint_params}})
    numbasiskinds = inputdata["numbasiskinds"]
    atomtypes = inputdata["atomtypes"]
    keys = Symbol.(atomtypes)

    models = make_multimodel_Flux(atomtypes, fingerprintparams, inputdata, numbasiskinds)

    return models
end

function make_multimodel_Flux(atomtypes, fingerprintparams, inputdata, numbasiskinds)
    #model_total = BPChain_Lux_atom[]# NTuple{numbasiskinds,Lux.Chain} BPnet{numbasiskinds,Flux.Chain}[]
    model_total = Flux.Parallel[]# NTuple{numbasiskinds,Flux.Chain} BPnet{numbasiskinds,FFlux.Chain}[]
    #model_total = Flux.SkipConnection[]
    #model_total = Flux.Chain[]#
    # model_total = NamedTuple[]# NTuple{numbasiskinds,Flux.Chain} BPnet{numbasiskinds,FFlux.Chain}[]

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
                    error("notsupported")
                    #model = make_kanmodel(layerstructure, activations, kantype, orders)
                else
                    model = make_densemodel_Flux(layerstructure, activations, resnet)
                end
            else
                layerstructure = zeros(Int64, 2)
                layerstructure[1] = numparams
                layerstructure[2] = 1
                activations = Vector{Any}(undef, 1)
                activations[end] = nothing
                model = Flux.Chain(Flux.Dense(numparams, 1))
            end

            #display(model)
            push!(models, model)
        end
        #if numbasiskinds > 1
        model_itype = Flux.Parallel(+, models...)
        #else
        #    model_itype = Flux.Parallel(x -> x, models...)
        #end
        if numbasiskinds > 1
            #    model_itype = Flux.Parallel(+, models...)
        else
            #    model_itype = Flux.Chain(models[1])
        end
        #keys_atom = Tuple(Symbol.(collect(1:numbasiskinds)))
        #model_itype = NamedTuple{keys_atom}(models)
        #model_itype = BPChain_Flux_atom(NamedTuple{keys_atom}(models)) #Tuple(models)
        display(model_itype)
        #model_parallel = Flux.Chain((x, y) -> model_itype(x) * y) #
        model_parallel = Flux.Parallel(*, model_itype, Flux.Chain(x -> x))
        #model_skip = Flux.SkipConnection(model_itype, (out1, x2) -> out1 * x2)
        #push!(model_total, model_skip)
        #push!(model_total, model_itype)
        push!(model_total, model_parallel)
    end

    #T = typeof(Tuple(model_total))
    keys = Tuple(Symbol.(atomtypes))
    #display(model_total)
    models = Flux.Parallel(+, model_total...)
    #models = NamedTuple{keys}(model_total)
    # models = BPChain_Flux{T}(Tuple(model_total))
    display(models)
    return BPChain_Flux(models)
end


function make_densemodel_Flux(layers, activations, resnet=false)
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
                    d = Flux.Parallel(+, x -> x, Flux.Dense(istart, iend, activations[i]))
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

function (l::BPChain_Flux)(x)#, ps, st::NamedTuple)
    energies = l.layers(x)#Lux.apply(l.layers, x, ps, st)
    #l.layers(x..., ps, st)
    return energies
end