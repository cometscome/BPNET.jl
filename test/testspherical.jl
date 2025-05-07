
using BPNET
using Test

using Downloads
using CodecBzip2
using Tar
using TranscodingStreams
using Flux
using MLUtils



function trainingtest()
    tomlfile = "input.toml"
    bpdata, inputdata = BP_dataset_fromTOML(tomlfile)
    ratio = inputdata["testratio"]
    filename_train = inputdata["filename_train"]
    filename_test = inputdata["filename_test"]
    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)
    traindata = BPdata_memory(bpdata, filename_train)
    #return

    numbatch = inputdata["numbatch"]
    train_loader = DataLoader(traindata; batchsize=numbatch)
    println("num. of training data $(length(traindata))")

    testdata = BPdata_memory(bpdata, filename_test)
    test_loader = DataLoader(testdata; batchsize=1)
    println("num. of testing data $(length(testdata))")

    model = BPnet(inputdata, bpdata.fingerprint_parameters) |> f64
    x, y = traindata[1]
    e = model(x)
    println(e)

    display(model)

    θ, re = Flux.destructure(model)
    #grad = Flux.gradient(θ -> sum(re(θ)(x)), θ)
    #display(grad[1])

    if inputdata["gpu"]
        θ = fmap(CuArray{Float64}, θ)
    end

    state = set_state(inputdata, θ)
    lossfunction(x, y) = Flux.mse(x, y)
    # println(lossfunction(x, y))
    println("num. of parameters: $(length(θ))")


    nepoch = inputdata["nepoch"]
    training!(θ, re, state, train_loader, test_loader, lossfunction, nepoch; modelparamfile=inputdata["modelparamfile"])

end

trainingtest()