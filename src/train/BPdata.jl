struct BPdata{keys,num_of_types,num_of_structs,T,numbasiskinds}
    fp::T
    E_scale::Float64
    E_shift::Float64
end

function BPdata(dataset::BP_dataset{keys,num_of_types,num_of_structs,Tfp,Td,numbasiskinds}, filename) where {keys,num_of_types,num_of_structs,Tfp,Td,numbasiskinds}
    fp = jldopen(filename, "r")
    num = Int(fp["num_of_structs"])
    T = typeof(fp)
    return BPdata{keys,num_of_types,num,T,numbasiskinds}(fp, dataset.E_scale, dataset.E_shift)
end
export BPdata

function get_coeff(data::BPdata{keys,num_of_types,num_of_structs,T,1}, istruct) where {keys,num_of_types,num_of_structs,T}
    fp = data.fp
    energy = 0.0
    coefficients = Vector{Matrix{Float64}}(undef, num_of_types)
    natoms = 0
    #jldopen(dataset.datafilenames[istruct], "r") do file
    energy = fp["$istruct"]["energy"]
    coefficients .= fp["$istruct"]["coefficients"]
    natoms = fp["$istruct"]["natoms"]

    return energy, coefficients, natoms
end

function get_coeff(data::BPdata{keys,num_of_types,num_of_structs,T,numbasiskinds}, istruct) where {keys,num_of_types,num_of_structs,T,numbasiskinds}
    fp = data.fp
    energy = 0.0
    coefficients = Vector{Vector{Matrix{Float64}}}(undef, num_of_types)
    natoms = 0
    #jldopen(dataset.datafilenames[istruct], "r") do file
    energy = fp["$istruct"]["energy"]
    coefficients .= fp["$istruct"]["coefficients"]
    natoms = fp["$istruct"]["natoms"]

    return energy, coefficients, natoms
end

Base.length(dataset::BPdata{keys,num_of_types,num_of_structs,T,numbasiskinds}) where {keys,num_of_types,num_of_structs,T,numbasiskinds} = num_of_structs

function Base.getindex(dataset::BPdata, i::Int)
    return dataset[i:i]
end

function Base.getindex(dataset::BPdata{keys,num_of_types,num_of_structs,T,1},
    I::AbstractVector) where {keys,num_of_types,num_of_structs,T}
    num = length(I)

    coefficients_batch = Vector{Matrix{Float64}}(undef, num_of_types)
    structindices = Vector{Vector{Int64}}(undef, num_of_types)
    for itype = 1:num_of_types
        structindices[itype] = Int64[]
    end
    coefficients_batch = Vector{Matrix{Float64}}(undef, num_of_types)
    energy_batch = zeros(Float64, 1, num)

    vec_coefficients = Vector{Vector{Matrix{Float64}}}(undef, num_of_types)
    for itype = 1:num_of_types
        vec_coefficients[itype] = Vector{Matrix{Float64}}(undef, num)
    end

    totalnumatom = 0
    for i = 1:num
        istruct = I[i]
        energy, coefficients, natoms = get_coeff(dataset, istruct)


        energy_batch[i] = energy
        for itype = 1:num_of_types
            num_parameters_itype, num_atoms_itype = size(coefficients[itype])
            totalnumatom += num_atoms_itype
            vec_coefficients[itype][i] = coefficients[itype]
            for iatom = 1:num_atoms_itype
                push!(structindices[itype], i)
            end
        end
    end


    for itype = 1:num_of_types
        coefficients_batch[itype] = hcat(vec_coefficients[itype]...)
    end


    #structindicesmatrix = Vector{SparseMatrixCSC{Int64,Int64}}(undef, num_of_types)
    structindicesmatrix = Vector{Matrix{Float64}}(undef, num_of_types)
    #structindicesmatrix = Vector{SparseMatrixCSC{Int64,Int64}}(undef, num_of_types)
    #structindicesmatrix = Vector{Matrix{Bool}}(undef, num_of_types)


    for itype = 1:num_of_types
        num_parameters_itype, num_atoms_itype = size(coefficients_batch[itype])
        indices_i = structindices[itype]#,#view(structindices[itype], 1:num)
        #structindicesmatrix[itype] = Matrix(sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)')
        #structindicesmatrix[itype] =sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)'
        structindicesmatrix[itype] = Matrix(sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)')

        #structindicesmatrix[itype] = sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)
    end

    data = NamedTuple{keys,NTuple{num_of_types,Matrix{Float64}}}(Tuple(coefficients_batch))
    #labels = NamedTuple{keys,NTuple{num_of_types,SparseMatrixCSC{Int64,Int64}}}(Tuple(structindicesmatrix))
    labels = NamedTuple{keys,NTuple{num_of_types,Matrix{Float64}}}(Tuple(structindicesmatrix))
    #labels = NamedTuple{keys,NTuple{num_of_types,Matrix{Bool}}}(Tuple(structindicesmatrix))

    xbatch = @NamedTuple{data::typeof(coefficients_batch[1]), labels::typeof(structindicesmatrix[1])}[]
    #xbatch = Tuple{typeof(coefficients_batch[1]),typeof(structindicesmatrix[1])}[]
    #return coefficients_batch,structindicesmatrix,energy_batch,num,totalnumatom


    for itype = 1:num_of_types
        push!(xbatch, (coefficients_batch[itype], structindicesmatrix[itype]))
    end
    return xbatch, energy_batch, num, totalnumatom

    #return (data=data, labels=labels), energy_batch, num, totalnumatom

    return (data=data, labels=labels, numdata=num, totalnumatom=totalnumatom), energy_batch

end




function Base.getindex(dataset::BPdata{keys,num_of_types,num_of_structs,T,numbasiskinds},
    I::AbstractVector) where {keys,num_of_types,num_of_structs,T,numbasiskinds}
    num = length(I)

    coefficients_batch = Vector{Vector{Matrix{Float64}}}(undef, num_of_types)
    for itype = 1:num_of_types
        coefficients_batch[itype] = Vector{Matrix{Float64}}(undef, numbasiskinds)
    end

    structindices = Vector{Vector{Int64}}(undef, num_of_types)
    for itype = 1:num_of_types
        structindices[itype] = Int64[]
    end
    energy_batch = zeros(Float64, 1, num)

    vec_coefficients = Vector{Vector{Vector{Matrix{Float64}}}}(undef, num_of_types)
    for itype = 1:num_of_types
        vec_coefficients[itype] = Vector{Vector{Matrix{Float64}}}(undef, numbasiskinds)
        for ikind = 1:numbasiskinds
            vec_coefficients[itype][ikind] = Vector{Vector{Matrix{Float64}}}(undef, num)
        end
    end

    totalnumatom = 0
    for i = 1:num
        istruct = I[i]
        energy, coefficients, natoms = get_coeff(dataset, istruct)


        energy_batch[i] = energy
        for itype = 1:num_of_types

            #for ikind = 1:numbasiskinds
            #    display(coefficients[itype][ikind])
            #end

            _, num_atoms_itype = size(coefficients[itype][1])
            totalnumatom += num_atoms_itype
            for ikind = 1:numbasiskinds
                vec_coefficients[itype][ikind][i] = coefficients[itype][ikind]
            end
            #vec_coefficients[itype][i] = coefficients[itype]
            for iatom = 1:num_atoms_itype
                push!(structindices[itype], i)
            end
        end
    end


    for itype = 1:num_of_types
        for ikind = 1:numbasiskinds
            coefficients_batch[itype][ikind] = hcat(vec_coefficients[itype][ikind]...)
        end
    end


    #structindicesmatrix = Vector{SparseMatrixCSC{Int64,Int64}}(undef, num_of_types)
    structindicesmatrix = Vector{Matrix{Float64}}(undef, num_of_types)
    #structindicesmatrix = Vector{SparseMatrixCSC{Int64,Int64}}(undef, num_of_types)
    #structindicesmatrix = Vector{Matrix{Bool}}(undef, num_of_types)


    for itype = 1:num_of_types
        num_parameters_itype, num_atoms_itype = size(coefficients_batch[itype][1])
        indices_i = structindices[itype]#,#view(structindices[itype], 1:num)
        #structindicesmatrix[itype] = Matrix(sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)')
        #structindicesmatrix[itype] =sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)'
        structindicesmatrix[itype] = Matrix(sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)')

        #structindicesmatrix[itype] = sparse(indices_i, 1:length(indices_i), 1, num, num_atoms_itype)
    end

    data = NamedTuple{keys,NTuple{num_of_types,Vector{Matrix{Float64}}}}(Tuple(coefficients_batch))
    #labels = NamedTuple{keys,NTuple{num_of_types,SparseMatrixCSC{Int64,Int64}}}(Tuple(structindicesmatrix))
    labels = NamedTuple{keys,NTuple{num_of_types,Matrix{Float64}}}(Tuple(structindicesmatrix))
    #labels = NamedTuple{keys,NTuple{num_of_types,Matrix{Bool}}}(Tuple(structindicesmatrix))

    xbatch = @NamedTuple{data::typeof(coefficients_batch[1]), labels::typeof(structindicesmatrix[1])}[]
    #xbatch = Tuple{typeof(coefficients_batch[1]),typeof(structindicesmatrix[1])}[]
    #return coefficients_batch,structindicesmatrix,energy_batch,num,totalnumatom


    for itype = 1:num_of_types
        push!(xbatch, (coefficients_batch[itype], structindicesmatrix[itype]))
    end
    return xbatch, energy_batch, num, totalnumatom

    #return (data=data, labels=labels), energy_batch, num, totalnumatom

    return (data=data, labels=labels, numdata=num, totalnumatom=totalnumatom), energy_batch

end

