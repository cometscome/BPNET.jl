using Lux
#struct BPChain_Lux{L<:NamedTuple} <: Lux.AbstractLuxWrapperLayer{:layers}
#    layers::L
#end
struct BPChain_Lux{L} <: Lux.AbstractLuxWrapperLayer{:layers}
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
    #model_total = BPChain_Lux_atom[]# NTuple{numbasiskinds,Lux.Chain} BPnet{numbasiskinds,Flux.Chain}[]
    model_total = Lux.Parallel[]# NTuple{numbasiskinds,Lux.Chain} BPnet{numbasiskinds,Flux.Chain}[]
    #model_total = Lux.SkipConnection[]
    #model_total = Lux.Chain[]#
    # model_total = NamedTuple[]# NTuple{numbasiskinds,Lux.Chain} BPnet{numbasiskinds,Flux.Chain}[]

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
        #if numbasiskinds > 1
        model_itype = Lux.Parallel(+, models...)
        #else
        #    model_itype = Lux.Parallel(x -> x, models...)
        #end
        if numbasiskinds > 1
            #    model_itype = Lux.Parallel(+, models...)
        else
            #    model_itype = Lux.Chain(models[1])
        end
        #keys_atom = Tuple(Symbol.(collect(1:numbasiskinds)))
        #model_itype = NamedTuple{keys_atom}(models)
        #model_itype = BPChain_Lux_atom(NamedTuple{keys_atom}(models)) #Tuple(models)
        display(model_itype)
        #model_parallel = Lux.Chain((x, y) -> model_itype(x) * y) #
        model_parallel = Lux.Parallel(*, model_itype, Lux.Chain(x -> x))
        #model_skip = Lux.SkipConnection(model_itype, (out1, x2) -> out1 * x2)
        #push!(model_total, model_skip)
        #push!(model_total, model_itype)
        push!(model_total, model_parallel)
    end

    #T = typeof(Tuple(model_total))
    keys = Tuple(Symbol.(atomtypes))
    #display(model_total)
    models = Lux.Parallel(+, model_total...)
    #models = NamedTuple{keys}(model_total)
    # models = BPChain_Lux{T}(Tuple(model_total))
    display(models)
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

using ComponentArrays

function (l::BPChain_Lux)(x, ps, st::NamedTuple)
    #@show typeof(x)
    #@show x
    #display(x)
    #display(x[1][1])
    #m1 = l.layers.layer_1.layer_1
    #m2 = l.layers.layer_1.layer_1
    #display(m1)
    #display(m2)

    #@code_warntype l.layers(x..., ps, st)
    energies, st = Lux.apply(l.layers, x, ps, st)
    #l.layers(x..., ps, st)
    return energies, st
    #println(values(ps))
    pskeys = keys(ps)
    i = 1
    name = pskeys[i]
    ps_i = ps[name]
    st_i = getfield(st, name)
    x_i = x[i]
    model_i = getfield(l.layers, name)
    energies = apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
    for i = 2:length(pskeys)
        name = pskeys[i]
        ps_i = ps[name]
        st_i = getfield(st, name)
        x_i = x[i]
        model_i = getfield(l.layers, name)
        energies += apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
    end
    return energies, st

    for (i, name) in enumerate(keys(ps))
        ps_i = ps[name]
        st_i = getfield(st, name)
        x_i = x[i]
        model_i = getfield(l.layers, name)
        println(typeof(ps))
        println(typeof(ps_i))
        #energy_i = apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
        if i == 1
            energies = apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
        else
            energies += apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
        end
        #energies = apply_bpmultimodel_Lux!(model_i, x_i, ps_i, st_i)
    end
    #energies = sum(map(apply_bpmultimodel_Lux!, l.layers, x, ps, st))

    return energies, st
end

function (l::BPChain_Lux_atom)(x, ps, st::NamedTuple)
    energies_i, _ = Lux.apply(l.layers[1], x[1], ps[1], st[1])
    for i = 2:length(l.layers)
        e, _ = Lux.apply(l.layers[i], x[i], ps[i], st[i])
        energies_i += e
    end
    return energies_i, st


    function f(model, x, ps, st)
        d, _ = Lux.apply(model, x, ps, st)
        return d
    end
    energies_i = sum(map(f, l.layers, x, ps, st))
    return energies_i, st
end

function apply_bpmultimodel_Lux!(model_i, x, ps_i, st_i)
    #display(model_i)

    #@show typeof(ps_i)

    #println(typeof(Tuple(x.data)))
    #display(x.data)
    energies_i, _ = Lux.apply(model_i, Tuple(x.data), ps_i, st_i)#model_i(x.data, ps_i, st_i)
    #display(energies_i)
    #error("oh")


    energies = energies_i * x.labels
    return energies

    function f(model, x, ps, st)
        d, _ = Lux.apply(model, x, ps, st)
        return d
    end
    #f(model, x) =
    energies_i = sum(map(f, model_i, x.data, ps_i, st_i))
    energies = energies_i * x.labels
    return energies
end




