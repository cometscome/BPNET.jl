struct BPdata_memory{keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA}
    E_scale::Float64
    E_shift::Float64
    energy_all::TE
    coefficients_all::TC
    natoms_all::TNA
end

function BPdata_memory(dataset::BP_dataset{keys,num_of_types,num_of_structs,Tfp,Td,numbasiskinds}, filename) where {keys,num_of_types,num_of_structs,Tfp,Td,numbasiskinds}
    fp = jldopen(filename, "r")
    num = Int(fp["num_of_structs"])
    energy_all = zeros(Float64, num)
    natoms_all = zeros(Int64, num)
    coefficients_all = Vector{Vector{Vector{Matrix{Float64}}}}(undef, num)
    if numbasiskinds == 1
        coefficients_all = Vector{Vector{Matrix{Float64}}}(undef, num)
    else
        coefficients_all = Vector{Vector{Vector{Matrix{Float64}}}}(undef, num)
    end

    for istruct = 1:num
        energy_all[istruct] = fp["$istruct"]["energy"]
        coefficients_all[istruct] = fp["$istruct"]["coefficients"]
        natoms_all[istruct] = fp["$istruct"]["natoms"]
    end
    TE = typeof(energy_all)
    TC = typeof(coefficients_all)
    TNA = typeof(natoms_all)

    return BPdata_memory{keys,num_of_types,num,numbasiskinds,TE,TC,TNA}(dataset.E_scale, dataset.E_shift,
        energy_all,
        coefficients_all,
        natoms_all
    )
end
export BPdata_memory

Base.length(dataset::BPdata_memory{keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA}) where {keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA} = num_of_structs


function Base.getindex(dataset::BPdata_memory, i::Int)
    return dataset[i:i]
end

function get_coeff(data::BPdata_memory{keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA},
    istruct) where {keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA}
    return data.energy_all[istruct], data.coefficients_all[istruct], data.natoms_all[istruct]#energy, coefficients, natoms
end


function Base.getindex(dataset::BPdata_memory{keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA},
    I::AbstractVector) where {keys,num_of_types,num_of_structs,numbasiskinds,TE,TC,TNA}
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

            if numbasiskinds == 1
                _, num_atoms_itype = size(coefficients[itype])
            else
                _, num_atoms_itype = size(coefficients[itype][1])
            end

            totalnumatom += num_atoms_itype
            if numbasiskinds == 1
                vec_coefficients[itype][1][i] = coefficients[itype]
            else
                for ikind = 1:numbasiskinds
                    vec_coefficients[itype][ikind][i] = coefficients[itype][ikind]
                end
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

    #println(I, "\t", typeof(coefficients_batch[1]))
    xbatch = @NamedTuple{data::typeof(coefficients_batch[1]), labels::typeof(structindicesmatrix[1])}[]
    #xbatch = Tuple{typeof(coefficients_batch[1]),typeof(structindicesmatrix[1])}[]
    #return coefficients_batch,structindicesmatrix,energy_batch,num,totalnumatom


    for itype = 1:num_of_types
        push!(xbatch, (data=coefficients_batch[itype], labels=structindicesmatrix[itype]))
    end
    return xbatch, energy_batch, num, totalnumatom

    #return (data=data, labels=labels), energy_batch, num, totalnumatom

    return (data=data, labels=labels, numdata=num, totalnumatom=totalnumatom), energy_batch

end
