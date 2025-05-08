function write_network_info(io::IO, nlayers, nnodesmax, Wsize, nvalues, nnodes, fun, iw, iv, W)
    println(io, nlayers)
    println(io, nnodesmax)
    println(io, Wsize)
    println(io, nvalues)
    println(io, join(nnodes, " "))
    println(io, join(fun, " "))
    println(io, join(iw, " "))
    println(io, join(iv, " "))
    println(io, join(W, " "))
end

function write_sf_setup(io::IO, description, atomtype, nenv, envtypes, rc_min, rc_max, sftype, nsf, nsfparam, sf, sfparam, sfenv, sfval_min, sfval_max, sfval_avg, sfval_cov, neval)
    println(io, description)
    println(io, atomtype)
    println(io, nenv)
    println(io, join(envtypes, " "))
    println(io, rc_min)
    println(io, rc_max)
    println(io, sftype)
    println(io, nsf)
    println(io, nsfparam)
    println(io, join(sf, " "))
    for i in 1:size(sfparam, 2)
        println(io, join(sfparam[:, i], " "))
    end
    for i in 1:size(sfenv, 2)
        println(io, join(sfenv[:, i], " "))
    end
    println(io, neval)
    println(io, join(sfval_min, " "))
    println(io, join(sfval_max, " "))
    println(io, join(sfval_avg, " "))
    println(io, join(sfval_cov, " "))
end

function write_trainset_info(io::IO, file, normalized, scale, shift, ntypes, type_names, E_atom, natomtot, nstrucs, E_min, E_max, E_avg)
    println(io, file)
    println(io, normalized)
    println(io, scale)
    println(io, shift)
    println(io, ntypes)
    println(io, join(type_names, " "))
    println(io, join(E_atom, " "))
    println(io, natomtot)
    println(io, nstrucs)
    println(io, E_min, " ", E_max, " ", E_avg)
end

function write_ascii_format(filename, data)
    open(filename, "w") do io
        write_network_info(io, data.nlayers, data.nnodesmax, data.Wsize, data.nvalues, data.nnodes, data.fun, data.iw, data.iv, data.W)
        write_sf_setup(io, data.description, data.atomtype, data.nenv, data.envtypes, data.rc_min, data.rc_max, data.sftype, data.nsf, data.nsfparam, data.sf, data.sfparam, data.sfenv, data.sfval_min, data.sfval_max, data.sfval_avg, data.sfval_cov, data.neval)
        write_trainset_info(io, data.file, data.normalized, data.scale, data.shift, data.ntypes, data.type_names, data.E_atom, data.natomtot, data.nstrucs, data.E_min, data.E_max, data.E_avg)
    end
end

export write_ascii_format