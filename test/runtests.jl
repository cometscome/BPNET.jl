using BPNET
using Test

using Downloads
using CodecBzip2
using Tar
using TranscodingStreams
using Flux
using MLUtils
using Optimisers
using Random
#ENV["JULIA_ENZYME_PRINT_ACTIVITY"] = "1"
import Enzyme
#using Enzyme: Const, Duplicated
using LinearAlgebra
using InteractiveUtils





function downloadtest()
    # URL of the file
    url = "http://ann.atomistic.net/files/aenet-example-02-TiO2-Chebyshev.tar.bz2"

    # Download the file
    filename = "aenet-example-02-TiO2-Chebyshev.tar.bz2"
    Downloads.download(url, filename)

    # Decompress the .bz2 file to a .tar file
    tar_filename = replace(filename, ".bz2" => "")
    open(filename, "r") do input_file
        open(tar_filename, "w") do output_file
            stream = TranscodingStreams.TranscodingStream(CodecBzip2.Bzip2Decompressor(), input_file)
            write(output_file, stream)
        end
    end

    # Ensure the directory is empty or create it
    extract_dir = "extracted_files"
    if isdir(extract_dir)
        rm(extract_dir; recursive=true)  # Remove existing directory and its contents
    end
    mkdir(extract_dir)

    # Extract to the new directory
    Tar.extract(tar_filename, extract_dir)

    # Delete the .tar and .tar.bz2 files after extraction
    rm(filename)
    rm(tar_filename)

    println("Download, decompression, extraction, and cleanup completed. Files are in '$extract_dir'.")
end

function generatetest()
    envtypes = ["Ti", "O"]
    g = Generator(envtypes)


    atomtype = "Ti"
    f1 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f1)

    atomtype = "O"
    f2 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f2)



    exampledir = "extracted_files/aenet-example-02-TiO2-Chebyshev/TiO2-xsf"
    datafiles = filter(x -> x[end-3:end] == ".xsf", readdir(exampledir, join=true))
    adddata!(g, datafiles)
    set_numfiles!(g, 1000)


    display(g.numfiles)

    outputfile = make_descriptor(g)
    println(outputfile)

    return

    filename = make_generatein(g)
    make_fingerprintfile(g)

    generate(filename)

end

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

    @code_warntype model(x)
    return

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

function generatebasistest()
    envtypes = ["Ti", "O"]
    g = Generator(envtypes)


    atomtype = "Ti"
    f1 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f1)

    atomtype = "O"
    f2 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f2)



    exampledir = "extracted_files/aenet-example-02-TiO2-Chebyshev/TiO2-xsf"
    datafiles = filter(x -> x[end-3:end] == ".xsf", readdir(exampledir, join=true))
    adddata!(g, datafiles)
    set_numfiles!(g, 1000)

    filename = get_filename(g, 1)
    println(filename)
    nc_R = 10
    nc_θ = 4
    Rmin = 0.75
    Rc = 8

    clayer = ChebyshevLayer(; nc_R, nc_θ, Rmin, Rc)
    display(clayer)

    xsffilelist = get_filenames(g)
    kind_list = envtypes
    generate_descriptors(clayer, xsffilelist, kind_list; Rmax=Rc)


end


function testfunc(R, R_i, R_js, kinds, numatoms, cell, kanbasis, modellayer)
    c = kanbasis(kinds, numatoms, R, cell, R_i, R_js)
    c1 = c[1]
    s = sum(c1) #modellayer[1](c[1])
    return sum(s)
    #return sum(Dense(size(c[1], 1), 1)(c[1]))  # 一部だけ使う
end

function testfunction2(R, R_i, R_js, kinds, numatoms, cell, modellayer, kanbasis)
    c = kanbasis(kinds, numatoms, R, cell, R_i, R_js)
    d1 = c[1]
    d2 = c[2]
    dd1 = sum(modellayer[1](d1))
    dd2 = sum(modellayer[2](d2))
    return sum(dd1) + sum(dd2)
    #display(c[1])
end

function testfunction3(R, R_i, R_js, kinds, numatoms, cell, pmodel, kanbasis)
    c = kanbasis(kinds, numatoms, R, cell, R_i, R_js)
    #d1 = c[1]
    #d2 = c[2]
    #dd1 = sum(modellayer[1](d1))
    #dd2 = sum(modellayer[2](d2))
    #return sum(dd1) + sum(dd2)
    #ct = Tuple(c)
    d = pmodel(c)
    return sum(sum.(d))

    #d1 = c[1]
    #d2 = c[2]
    #dd1 = sum(modellayer[1](d1))
    #dd2 = sum(modellayer[2](d2))
    #return sum(dd1) + sum(dd2)
    #display(c[1])
end

function testfunction4(xsf, R_i, R_js, pmodel, kanbasis)
    c = kanbasis(xsf, R_i, R_js)
    d = pmodel(c)
    return sum(sum.(d))
end


function testfunction5(R, xsfstatic, R_i, R_js, pmodel, kanbasis)
    c = kanbasis(R, xsfstatic, R_i, R_js)
    d = pmodel(c)
    return sum(sum.(d))
end

function testfunction6(Rs, xsfstatics, R_i, R_js, pmodel, kanbasis)
    c = kanbasis(Rs, xsfstatics, R_i, R_js)
    d = pmodel(c)
    return sum(sum.(d))
end

function testfunction7(R, xsfstatic, pmodel, kanbasis)
    c = kanbasis(R, xsfstatic)
    d = pmodel(c)
    return sum(sum.(d))
end

function testfunction8(Rs, xsfstatics, pmodel, kanbasis)
    c = kanbasis(Rs, xsfstatics)
    d = pmodel(c)
    return sum(sum.(d))
end

function testfunction9(Rs, xsfstatics, pos, pmodel, kanbasis)
    c = kanbasis(Rs, xsfstatics)
    d = pmodel(c)
    de = d .* pos
    #println(de)
    result = reduce((x, y) -> x .* y, de)
    return sum(result)
    #return sum(sum.(de))
end


function testfunction10(Rs, xsfstatics, pos, network)
    result = network(Rs, xsfstatics, pos)
    return sum(result)
    #return sum(sum.(de))
end



function xsftrainingtest()
    envtypes = ["Ti", "O"]
    #=
    Ti -1626.66972707  | eV
    O   -433.23448532  | eV
    =#
    isolated_energies = [-1626.66972707, -433.23448532]
    g = Generator(envtypes; isolated_energies)
    atomtype = "Ti"
    f1 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)
    push!(g, f1)

    atomtype = "O"
    f2 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)
    push!(g, f2)

    exampledir = "extracted_files/aenet-example-02-TiO2-Chebyshev/TiO2-xsf"
    datafiles = filter(x -> x[end-3:end] == ".xsf", readdir(exampledir, join=true))
    kind_list = envtypes

    isolated_energies = get_isolated_energies(g)
    display(isolated_energies)
    #return

    numfiles = length(datafiles)
    println("num. of training data $numfiles")
    indices = collect(1:numfiles)
    shuffle!(indices)
    ratio = 0.9
    numtrain = Int64(floor(ratio * numfiles))
    numtest = numfiles - numtrain
    println("num. of training data $numtrain")
    println("num. of testing data $numtest")
    datafiles_train = datafiles[indices[1:numtrain]]
    dataset_train = KAN_dataset(kind_list, datafiles_train, isolated_energies)

    energies_train, numatoms_train = get_energies_and_numatoms(dataset_train)
    energies_peratom_train = energies_train ./ numatoms_train
    #display(energies_peratom_train)
    E_max = maximum(energies_peratom_train)
    E_min = minimum(energies_peratom_train)

    datafiles_test = datafiles[indices[numtrain+1:end]]
    dataset_test = KAN_dataset(kind_list, datafiles_test, isolated_energies)

    energies_test, numatoms_test = get_energies_and_numatoms(dataset_test)
    energies_peratom_test = energies_test ./ numatoms_test
    #display(energies_peratom_test)

    E_max = max(E_max, maximum(energies_peratom_test))
    E_min = min(E_min, minimum(energies_peratom_test))

    maxenergy = 1.0
    E_max = min(E_max, maxenergy)

    E_scale = 2.0 / (E_max - E_min)
    E_shift = 0.5 * (E_max + E_min)

    println("E_max = $E_max E_min = $E_min E_scale = $E_scale E_shift = $E_shift")

    rescale_energies!(dataset_train, E_scale, E_shift)
    rescale_energies!(dataset_test, E_scale, E_shift)

    remove_highenergy_structures!(dataset_train)
    remove_highenergy_structures!(dataset_test)


    energies_train, numatoms_train = get_energies_and_numatoms(dataset_train)
    energies_peratom_train = energies_train ./ numatoms_train
    #display(energies_peratom_train)

    energies_test, numatoms_test = get_energies_and_numatoms(dataset_test)
    energies_peratom_test = energies_test ./ numatoms_test
    #display(energies_peratom_test)

    #return

    numbatch = 16
    train_loader = DataLoader(dataset_train; batchsize=numbatch)
    test_loader = DataLoader(dataset_test; batchsize=1)

    kanlayer_Ti = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)
    kanlayer_O = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)
    kanbasis = KANbasis((Ti=kanlayer_Ti, O=kanlayer_O))


    numhidden = 10
    modellayer = Chain[]
    for atomkind in envtypes
        numbasis = get_numbasis(kanbasis, atomkind)
        push!(modellayer, Chain(Dense(numbasis, numhidden, relu), Dense(numhidden, 1)) |> f64)
    end

    pmodel = Parallel(IdentityTupleLayer(), modellayer...)


    network = BNnetwork(kanbasis, pmodel)

    function lossfunction(ytilde, y)
        d = 0.0
        n = length(y)
        for i = 1:n
            d += (ytilde[i] - y[i])^2
        end
        return d / n
    end

    dup_network = Enzyme.Duplicated(network)

    opt_state = Flux.setup(Adam(), network)

    epoch = 0
    numepoch = 100
    totalbatch = length(train_loader)
    for epoch = 1:numepoch
        ibatch = 0
        for (pos, xsfs, energies) in train_loader
            ibatch += 1
            Rs, xsfstatics = splitR_xsf(xsfs)
            #loss = lossfunction(network(Rs, xsfstatics, pos), energies)
            result = Flux.withgradient((m, Rs, xsfstatics, pos, energies) -> lossfunction(m(Rs, xsfstatics, pos), energies),
                dup_network,
                Enzyme.Const(Rs),
                Enzyme.Const(xsfstatics),
                Enzyme.Const(pos),
                Enzyme.Const(energies))
            loss = result.val
            grad = result.grad
            println("ibatch $(ibatch)/$(totalbatch) loss = ", loss)
            Flux.update!(opt_state, network, grad[1])
        end

        losstest = 0.0
        for (pos_t, xsfs_t, energies_t) in test_loader
            Rs_t, xsfstatics_t = splitR_xsf(xsfs_t)
            losstest += lossfunction(network(Rs_t, xsfstatics_t, pos_t), energies_t)
        end
        println("epoch $epoch test loss = ", losstest / length(test_loader))

    end
end

function xsftest()

    envtypes = ["Ti", "O"]
    g = Generator(envtypes)


    atomtype = "Ti"
    f1 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f1)

    atomtype = "O"
    f2 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f2)



    exampledir = "extracted_files/aenet-example-02-TiO2-Chebyshev/TiO2-xsf"
    datafiles = filter(x -> x[end-3:end] == ".xsf", readdir(exampledir, join=true))





    kind_list = envtypes

    #dataset = KAN_dataset(kind_list, datafiles)
    #numbatch = 10
    #train_loader = DataLoader(dataset, batchsize=numbatch)

    numfiles = length(datafiles)
    println("num. of training data $numfiles")
    indices = collect(1:numfiles)
    shuffle!(indices)
    #display(indices)
    ratio = 0.9
    numtrain = Int64(floor(ratio * numfiles))
    numtest = numfiles - numtrain
    println("num. of training data $numtrain")
    println("num. of testing data $numtest")
    datafiles_train = datafiles[indices[1:numtrain]]
    dataset_train = KAN_dataset(kind_list, datafiles_train)
    datafiles_test = datafiles[indices[numtrain+1:end]]
    dataset_test = KAN_dataset(kind_list, datafiles_test)

    numbatch = 10
    train_loader = DataLoader(dataset_train; batchsize=numbatch)
    test_loader = DataLoader(dataset_test; batchsize=1)


    #for (pos, xsfs, energy_batch) in train_loader
    #    display(energy_batch)
    #end
    #return
    pos, xsfs, energy_batch = dataset_train[1:3]
    Rmax = 8

    xsf = xsfs[1]
    display(xsf)
    ith_atom = 1
    #@code_llvm get_atoms_inside_the_sphere(xsf, ith_atom, basis.Rc)
    #error("dd")
    numatoms = BPNET.XSFreader.get_number(xsf)
    R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(
        numatoms, ith_atom, Rmax, xsf.R, xsf.kinds, xsf.cell)

    #testfunction = (numatoms, ith_atom, Rmax, R, kinds, cell) -> begin
    #    R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j = get_atoms_inside_the_sphere(
    #        numatoms, ith_atom, Rmax, R, kinds, cell)
    #    sum(norm(R_js))^2
    #end


    kanlayer_Ti = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)
    kanlayer_O = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)

    ith_atom = 1
    basis = kanlayer_Ti.basis
    radialbasis = KACbasis_radial(; polynomial_order=10, Rmin=0.75, Rc=8)
    clayer = ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8)
    angularbasis = KACbasis_angular(; polynomial_order=10, Rmin=0.75, Rc=8)

    kanbasis = KANbasis((Ti=kanlayer_Ti, O=kanlayer_O))

    #KANbasis_forward(numkinds, xsf.kinds, x.kanlayers, xsf.numatoms, xsf.R, xsf.cell, x.kind_list)
    #kanbasis(xsf.kinds, xsf.numatoms, xsf.R, xsf.cell, R_i, R_js)

    #c = kanbasis(xsf)
    #display(c)

    nmax = 300
    R_js = zeros(3, nmax)
    R_i = zeros(3)

    c = kanbasis(xsf.kinds, numatoms, xsf.R, xsf.cell, R_i, R_js)
    #display(c)
    #return

    #=
    testfunction2 = (R, R_i, R_js) -> begin
        c = kanbasis(xsf.kinds, numatoms, R, xsf.cell, R_i, R_js)
        sum(sum.(c))
    end
    =#

    #=
    function testfunction2(R, R_i, R_js, kinds, numatoms, cell)#, kanbasis)
        envtypes = ["Ti", "O"]
        kanlayer_Ti = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)
        kanlayer_O = KAN_Layer(ChebyshevLayer(; nc_R=10, nc_θ=4, Rmin=0.75, Rc=8), envtypes)
        kanbasis = KANbasis((Ti=kanlayer_Ti, O=kanlayer_O))

        c = kanbasis(kinds, numatoms, R, cell, R_i, R_js)
        #return sum(R)
        return sum(c)
    end
    =#


    #testfunction2(R, R_i, R_js) = sum(R)

    numhidden = 10
    modellayer = Chain[]
    for atomkind in envtypes
        numbasis = get_numbasis(kanbasis, atomkind)
        push!(modellayer, Chain(Dense(numbasis, numhidden, relu), Dense(numhidden, 1)) |> f64)
    end

    pmodel = Parallel(IdentityTupleLayer(), modellayer...)




    c = kanbasis(xsf.kinds, xsf.numatoms, xsf.R, xsf.cell, R_i, R_js)
    #display(c)
    #@code_warntype kanbasis(xsf.kinds, numatoms, xsf.R, xsf.cell, R_i, R_js)
    #@code_warntype testfunc(xsf.R, R_i, R_js, xsf.kinds, xsf.numatoms, xsf.cell, kanbasis, modellayer)

    #return

    dR = Enzyme.make_zero(xsf.R)
    #=
    Enzyme.autodiff(Enzyme.Reverse,
        testfunc,
        Enzyme.Active,
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Const(R_i),
        Enzyme.Const(R_js),
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(xsf.numatoms),
        Enzyme.Const(xsf.cell),
        Enzyme.Const(kanbasis),
        Enzyme.Const(modellayer)
    )
    display(dR)

    return
    =#

    #ENV["JULIA_ENZYME_LOG_LLVM"] = "0"

    #=
    function testfunction2(R, R_i, R_js)
        c = kanbasis(xsf.kinds, numatoms, R, xsf.cell, R_i, R_js)
        d1 = c[1]
        d2 = c[2]
        dd1 = sum(modellayer[1](d1))
        dd2 = modellayer[2](d2)
        return sum(dd1) #+ sum(dd2)
        #display(c[1])
        #return sum(sum.(c))
        #=
        d1 = modellayer[1](c[1])
        d2 = modellayer[2](c[2])
        return sum(d1) + sum(d2)
        d1 = sum(Dense(size(c[1])[1], 1)(c[1]))
        d2 = sum(Dense(size(c[2])[1], 1)(c[2]))
        return d1 + d2
        #sum(d)
        #d = pmodel(c...)
        =#
    end

    =#

    #=
    function testfunction2(c)
        #c = kanbasis(xsf.kinds, numatoms, R, xsf.cell, R_i, R_js)
        #display(c[1])
        d1 = sum(Dense(size(c[1])[1], 1)(c[1]))
        d2 = sum(Dense(size(c[2])[1], 1)(c[2]))
        return d1 + d2
        #sum(d)
        #d = pmodel(c...)
    end
    =#



    network = BNnetwork(kanbasis, pmodel)


    dR_n = Enzyme.make_zero(xsf.R)

    dh = 1e-8

    R, xsfstatic = splitR_xsf(xsf)

    Rs, xsfstatics = splitR_xsf(xsfs)
    dRs_n = Enzyme.make_zero(Rs)

    energies = reshape(get_energy.(xsfstatics), 1, :)
    function lossfunction(ytilde, y)
        d = 0.0
        n = length(y)
        for i = 1:n
            d += (ytilde[i] - y[i])^2
        end
        return d / n
    end
    #lossfunction(x, y) = Flux.mse(x, y)
    a = lossfunction(network(Rs, xsfstatics, pos), energies)


    @code_warntype lossfunction(network(Rs, xsfstatics, pos), energies)


    display(network)

    dup_network = Enzyme.Duplicated(network)



    θ, rebuild = Flux.destructure(network)
    #return
    #return

    #value0 = testfunction2(xsf.R, R_i, R_js, xsf.kinds, xsf.numatoms, xsf.cell, modellayer, kanbasis)
    #value0 = testfunction4(xsf, R_i, R_js, pmodel, kanbasis)
    #value0 = testfunction5(R, xsfstatic, R_i, R_js, pmodel, kanbasis)
    #value0 = testfunction6(Rs, xsfstatics, R_i, R_js, pmodel, kanbasis)
    #value0 = testfunction7(R, xsfstatic, pmodel, kanbasis)
    #value0 = testfunction8(Rs, xsfstatics, pmodel, kanbasis)
    #value0 = testfunction9(Rs, xsfstatics, pos, pmodel, kanbasis)
    #value0 = testfunction10(Rs, xsfstatics, pos, network)
    #value0 = network(Rs, xsfstatics, pos)
    value0 = rebuild(θ)(Rs, xsfstatics, pos, energies)



    #value0 = testfunction5(R, xsfstatic, R_i, R_js, pmodel, kanbasis)
    #value0 = testfunction3(xsf.R, R_i, R_js, xsf.kinds, xsf.numatoms, xsf.cell, pmodel, kanbasis)

    #value0 = testfunction2(xsf.R, R_i, R_js, xsf.kinds, numatoms, xsf.cell)#, kanbasis)#numatoms, ith_atom, Rmax, xsf.R, xsf.kinds, xsf.cell)
    #value0 = kanbasis(xsf.kinds, numatoms, xsf.R, xsf.cell, R_i, R_js)
    #@code_warntype testfunction4(xsf, R_i, R_js, pmodel, kanbasis)
    #@code_warntype testfunction5(R, xsfstatic, R_i, R_js, pmodel, kanbasis)
    #@code_warntype testfunction6(Rs, xsfstatics, R_i, R_js, pmodel, kanbasis)
    #@code_warntype testfunction7(R, xsfstatic, pmodel, kanbasis)
    #@code_warntype testfunction8(Rs, xsfstatics, pmodel, kanbasis)
    #@code_warntype testfunction9(Rs, xsfstatics, pos, pmodel, kanbasis)
    #@code_warntype testfunction10(Rs, xsfstatics, pos, network)
    #@code_warntype network(Rs, xsfstatics, pos)
    @code_warntype rebuild(θ)(Rs, xsfstatics, pos, energies)

    function make_thetaderivative(θ, rebuild, Rs, xsfstatics, pos, y; dh=1e-4)
        dθ = Enzyme.make_zero(θ)
        value0 = lossfunction(rebuild(θ)(Rs, xsfstatics, pos), y)
        for i = 1:length(θ)
            θd = deepcopy(θ)
            θd[i] += dh
            value = lossfunction(rebuild(θd)(Rs, xsfstatics, pos), y)
            dθ[i] = (value - value0) / dh
        end
        return dθ
    end

    dθ_n = make_thetaderivative(θ, rebuild, Rs, xsfstatics, pos, energies)
    display(dθ_n)
    Enzyme.API.printactivity!(false)
    Enzyme.API.printall!(false)

    grads_f = Flux.gradient((m, Rs, xsfstatics, pos, energies) -> lossfunction(m(Rs, xsfstatics, pos), energies),
        dup_network, Enzyme.Const(Rs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
        Enzyme.Const(energies))

    display(grads_f)
    return

    grads_f = Flux.gradient((m, Rs, xsfstatics, pos, energies) -> m(Rs, xsfstatics, pos, energies),
        dup_network, Enzyme.Const(Rs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
        Enzyme.Const(energies))

    display(grads_f)

    return

    rev0 = Enzyme.Reverse
    rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
    dθ = Enzyme.make_zero(θ)
    dRs = Enzyme.make_zero(Rs)

    Enzyme.autodiff(rev0,
        Enzyme.Const((θ, Rs, xsfstatics, pos) -> rebuild(θ)(Rs, xsfstatics, pos)),
        Enzyme.Active,
        Enzyme.Duplicated(θ, dθ),
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
    )
    display(dθ)
    return


    display(value0)

    for ixsf = 1:length(xsfstatics)
        nk, ni = size(Rs[ixsf])
        for i = 1:ni
            for k = 1:nk
                #R_n = deepcopy(R)
                Rs_n = deepcopy(Rs)
                #xsf_n = deepcopy(xsf)
                #R_n[k, i] = R[k, i] + dh
                Rs_n[ixsf][k, i] = Rs[ixsf][k, i] + dh
                #value = kanbasis(xsf.kinds, numatoms, R_n, xsf.cell, R_i, R_js) #numatoms, ith_atom, Rmax, R_n, xsf.kinds, xsf.cell)
                #value = testfunction2(R_n, R_i, R_js, xsf.kinds, numatoms, xsf.cell)#, kanbasis) #numatoms, ith_atom, Rmax, R_n, xsf.kinds, xsf.cell)
                #value = testfunction2(R_n, R_i, R_js, xsf.kinds, xsf.numatoms, xsf.cell, modellayer, kanbasis)
                #value = testfunction3(R_n, R_i, R_js, xsf.kinds, xsf.numatoms, xsf.cell, pmodel, kanbasis)
                #value = testfunction4(xsf_n, R_i, R_js, pmodel, kanbasis)
                #value = testfunction5(R_n, xsfstatic, R_i, R_js, pmodel, kanbasis)
                #value = testfunction6(Rs_n, xsfstatics, R_i, R_js, pmodel, kanbasis)
                #value = testfunction7(R_n, xsfstatic, pmodel, kanbasis)
                #value = testfunction8(Rs_n, xsfstatics, pmodel, kanbasis)
                #value = testfunction9(Rs_n, xsfstatics, pos, pmodel, kanbasis)
                value = rebuild(θ)(Rs_n, xsfstatics, pos)
                #value = network(Rs_n, xsfstatics, pos)
                #value = testfunction10(Rs_n, xsfstatics, pos, network)

                println("ixsf = ", ixsf, " i = ", i, " k = ", k)
                #display(value)
                #dR_n[k, i] = (value - value0) / dh
                dRs_n[ixsf][k, i] = (value - value0) / dh
            end
        end
    end
    #display(dR_n)
    display(dRs_n)

    #dR = Enzyme.make_zero(R)
    dRs = Enzyme.make_zero(Rs)
    #dxsf = Enzyme.make_zero(xsf)

    rev0 = Enzyme.Reverse
    rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
    dR_js = Enzyme.make_zero(R_js)
    dR_i = Enzyme.make_zero(R_i)
    #display(R_js)
    #return

    #=
    Enzyme.autodiff(rev0,
        #testfunction2,
        Enzyme.Const(testfunction2),
        Enzyme.Active,
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(numatoms),
        Enzyme.Const(xsf.cell))#,
    #        Enzyme.Active(kanbasis))
    #=    =#

    c = kanbasis(xsf.kinds, numatoms, xsf.R, xsf.cell, R_i, R_js)
    dc = Enzyme.make_zero(c)
    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction2),
        Enzyme.Active,
        Enzyme.Duplicated(c, dc)
    )
    display(dc)

    return
    =#

    Enzyme.autodiff(rev0,
        Enzyme.Const(network),
        Enzyme.Active,
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
    )
    display(dRs)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction10),
        Enzyme.Active,
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
        Enzyme.Const(network)
    )
    display(dRs)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction9),
        Enzyme.Active,
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pos),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dRs)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction8),
        Enzyme.Active,
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dRs)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction7),
        Enzyme.Active,
        Enzyme.Duplicated(R, dR),
        Enzyme.Const(xsfstatic),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dR)
    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction6),
        Enzyme.Active,
        Enzyme.Duplicated(Rs, dRs),
        Enzyme.Const(xsfstatics),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dRs)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction5),
        Enzyme.Active,
        Enzyme.Duplicated(R, dR),
        Enzyme.Const(xsfstatic),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dR)
    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction4),
        Enzyme.Active,
        Enzyme.Duplicated(xsf, dxsf),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dxsf.R)
    display(dxsf)
    #display(dR)

    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction3),
        Enzyme.Active,
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(xsf.numatoms),
        Enzyme.Const(xsf.cell),
        Enzyme.Const(pmodel),
        Enzyme.Const(kanbasis)
    )
    display(dR)
    return

    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction2),
        Enzyme.Active,
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(xsf.numatoms),
        Enzyme.Const(xsf.cell),
        Enzyme.Const(modellayer),
        Enzyme.Const(kanbasis)
    )
    #, xsf.kinds, xsf.numatoms, xsf.cell, modellayer, kanbasis)

    display(dR)


    return



    Enzyme.autodiff(rev0,
        Enzyme.Const(kanbasis),
        Enzyme.Active,
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(numatoms),
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Const(xsf.cell),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js),
    )


    display(dR)


    return

    testfunction = (R, R_i, R_js) -> begin
        #R_js = Vector{Float64}[]
        #for i = 1:numatoms
        #    for k = 1:3
        ##        R_js[k, i] = R[k, i]^2
        #    end
        #    #push!(R_js, R_js_i)
        #end
        atomkind_i, index_i, atomkinds_j, indices_j, count = get_atoms_inside_the_sphere!(
            numatoms, ith_atom, Rmax, R, xsf.kinds, xsf.cell, R_i, R_js)
        sum(norm(R_js))
    end

    nmax = 300
    R_js = zeros(3, nmax)
    R_i = zeros(3)
    dR_n = Enzyme.make_zero(xsf.R)
    dh = 1e-8
    value0 = testfunction(xsf.R, R_i, R_js)#numatoms, ith_atom, Rmax, xsf.R, xsf.kinds, xsf.cell)
    display(value0)
    nk, ni = size(xsf.R)
    for i = 1:ni
        for k = 1:nk
            R_n = deepcopy(xsf.R)
            R_n[k, i] = xsf.R[k, i] + dh
            value = testfunction(R_n, R_i, R_js) #numatoms, ith_atom, Rmax, R_n, xsf.kinds, xsf.cell)
            display(value)
            dR_n[k, i] = (value - value0) / dh
        end
    end
    display(dR_n)



    dR = Enzyme.make_zero(xsf.R)

    rev0 = Enzyme.Reverse
    rev = Enzyme.set_runtime_activity(Enzyme.Reverse)

    #=
    Enzyme.autodiff(rev0,
        Enzyme.Const(testfunction),
        Enzyme.Active,
        Enzyme.Const(numatoms),
        Enzyme.Const(ith_atom),
        Enzyme.Const(Rmax),
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Const(xsf.kinds),
        Enzyme.Const(xsf.cell))
        =#
    #display(dR)
    dR_js = Enzyme.make_zero(R_js)
    dR_i = Enzyme.make_zero(R_i)



    Enzyme.autodiff(rev,
        Enzyme.Const(testfunction),
        Enzyme.Active,
        Enzyme.Duplicated(xsf.R, dR),
        Enzyme.Duplicated(R_i, dR_i),
        Enzyme.Duplicated(R_js, dR_js))
    display(dR)


end

@testset "BPNET.jl" begin
    trainingtest()
    # Write your tests here.
    #downloadtest()
    #generatetest()
    #
    #generatetest()
    #trainingtest()
    #generatebasistest()
    #layertest()
    #println("done")
    #xsftrainingtest()
    #xsftest()
end
