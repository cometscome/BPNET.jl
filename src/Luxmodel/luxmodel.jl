using Lux
struct BPChain_Lux{L<:NamedTuple} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::L
end
export BPChain_Lux

struct BPChain_Lux_atom{L<:NamedTuple} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::L
end

Base.length(b::BPChain_Lux_atom) = length(b.layers)
Base.length(b::BPChain_Lux) = length(b.layers)

function BPChain_Lux(inputdata, fingerprintparams::Vector{Vector{FingerPrint_params}})
    numbasiskinds = inputdata["numbasiskinds"]
    atomtypes = inputdata["atomtypes"]
    keys = Symbol.(atomtypes)

    models = make_multimodel_Lux(atomtypes, fingerprintparams, inputdata, numbasiskinds)

    return models
end

function make_multimodel_Lux(atomtypes, fingerprintparams, inputdata, numbasiskinds)
    model_total = NamedTuple[]# NTuple{numbasiskinds,Lux.Chain} BPnet{numbasiskinds,Flux.Chain}[]
    kan = inputdata["kan"]
    resnet = inputdata["resnet"]

    for (itype, name) in enumerate(atomtypes)
        fingerprint_param = fingerprintparams[itype]

        models = Lux.Chain[]
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
                    model = make_densemodel_Lux(layerstructure, activations, resnet)
                end
            else
                layerstructure = zeros(Int64, 2)
                layerstructure[1] = numparams
                layerstructure[2] = 1
                activations = Vector{Any}(undef, 1)
                activations[end] = nothing
                model = Lux.Chain(Lux.Dense(numparams, 1))
            end

            #display(model)
            push!(models, model)
        end

        keys_atom = Tuple(Symbol.(collect(1:numbasiskinds)))
        model_itype = NamedTuple{keys_atom}(models)
        #model_itype = BPChain_Lux_atom(NamedTuple{keys_atom}(models)) #Tuple(models)
        display(model_itype)

        push!(model_total, model_itype)
    end

    #T = typeof(Tuple(model_total))
    keys = Tuple(Symbol.(atomtypes))
    #display(model_total)
    models = NamedTuple{keys}(model_total)
    # models = BPChain_Lux{T}(Tuple(model_total))
    #display(models)
    return BPChain_Lux(models)
end


function make_densemodel_Lux(layers, activations, resnet=false)
    #layers = [2,10,10,3]
    numlayers = length(layers) - 1
    modellist = []
    for i = 1:numlayers
        istart = layers[i]
        iend = layers[i+1]
        if activations[i] == nothing
            d = Lux.Dense(istart, iend)
        else
            if istart == iend
                if resnet
                    d = Lux.Parallel(+, x -> x, Lux.Dense(istart, iend, activations[i]))
                else
                    d = Lux.Dense(istart, iend, activations[i])
                end
            else
                d = Lux.Dense(istart, iend, activations[i])
            end
        end

        push!(modellist, d)
    end
    return Lux.Chain(modellist...)
end


function (l::BPChain_Lux)(x, ps, st::NamedTuple)


    #f(model_i, x, ps) = apply_bpmultimodel_Lux!(model_i, x, ps, st)
    energies = sum(map(apply_bpmultimodel_Lux!, l.layers, x, ps, st))

    return energies, st
end

function apply_bpmultimodel_Lux!(model_i, x, ps_i, st_i)


    function f(model, x, ps, st)
        d, _ = Lux.apply(model, x, ps, st)
        return d
    end
    #f(model, x) =
    energies_i = sum(map(f, model_i, x.data, ps_i, st_i))
    energies = energies_i * x.labels
    return energies
end




