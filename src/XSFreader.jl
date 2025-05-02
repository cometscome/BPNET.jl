module XSFreader
using LinearAlgebra
export XSFdata, get_energy, get_atoms_inside_the_sphere, make_Rmatrix, get_number, get_localRvectors
# Write your package code here.


struct LocalRdata
    R_i::Vector{Float64}
    atomkind_i::String
    index_i::Int64
    R_js::Vector{Vector{Float64}}
    atomkinds_j::Vector{String}
    indices_j::Vector{Int64}
end

mutable struct XSFdata
    R::Matrix{Float64} #3 x numatoms
    const F::Matrix{Float64} #3 x numatoms
    const comments::String
    const filename::String
    const cell::Matrix{Float64}
    const kinds::Vector{String}
    const species::Vector{String}
    const numatoms::Int64
    #haslocalinfo::Vector{Bool}
    #localinfo::Vector{Union{Nothing,LocalRdata}}
    #haslocalRvectors::Vector{Bool}
    #localRvectors::Vector{Union{Nothing,Matrix{Float64}}}
end

struct XSFdata_staticfields
    F::Matrix{Float64} #3 x numatoms
    comments::String
    filename::String
    cell::Matrix{Float64}
    kinds::Vector{String}
    species::Vector{String}
    numatoms::Int64
end
export XSFdata_staticfields

function splitR_xsf(xsf::XSFdata)
    xsfdata_static = XSFdata_staticfields(xsf.F,
        xsf.comments, xsf.filename,
        xsf.cell, xsf.kinds, xsf.species,
        xsf.numatoms)
    return xsf.R, xsfdata_static
end
function splitR_xsf(xsfs::AbstractVector{XSFdata})
    numfiles = length(xsfs)
    Rs = Vector{Matrix{Float64}}(undef, numfiles)
    xsfdata_statics = Vector{XSFdata_staticfields}(undef, numfiles)
    for ifile = 1:numfiles
        xsf = xsfs[ifile]
        Rs[ifile] = deepcopy(xsf.R)
        xsfdata_statics[ifile] = XSFdata_staticfields(xsf.F,
            xsf.comments, xsf.filename,
            xsf.cell, xsf.kinds, xsf.species,
            xsf.numatoms)
    end
    #println(Rs)
    return Rs, xsfdata_statics
end
export splitR_xsf

function get_energy(xsf::XSFdata)
    return parse(Float64, split(xsf.comments)[5])
end

function get_energy(xsf::XSFdata_staticfields)
    return parse(Float64, split(xsf.comments)[5])
end


function get_number(xsf::XSFdata)
    return xsf.numatoms
end
export get_number

function get_number_kind(xsf::XSFdata, string_ikind)
    num = 0
    for i = 1:xsf.numatoms
        num += ifelse(xsf.kinds[i] == string_ikind, 1, 0)
    end
    return num
end
export get_number_kind

function get_number_each(xsf::XSFdata, type_kinds::Vector{String})
    numkinds = length(type_kinds)
    nums = zeros(Int64, numkinds)
    for i = 1:xsf.numatoms
        ikind_i = findfirst(x -> x == xsf.kinds[i], type_kinds)
        nums[ikind_i] += 1
    end
    return nums
end
export get_number_each

function Base.display(xsf::XSFdata)
    numatoms = xsf.numatoms
    println(xsf.comments)
    println("unit cell:")
    display(xsf.cell)
    println("\t")
    println("----------------------------------------------------------------")
    println("num. of atoms: $(numatoms)")
    println("kind \t | X \t | Y \t | Z \t | F_x \t | F_y \t | F_z \t |")
    for i = 1:numatoms
        R = xsf.R[:, i]
        F = xsf.F[:, i]
        println("$(xsf.kinds[i]) \t $(R[1]) \t $(R[2]) \t $(R[3]) \t $(F[1]) \t $(F[2]) \t $(F[3])")
    end
end

function get_position(string, data)
    idata = 1
    for i = 1:length(data)
        u = split(data[i])
        if length(u) != 0
            if u[1] == string
                idata = i
                break
            end
        end
        if i == length(data)
            error("$string  is not found")
        end
    end
    return idata
end



function XSFdata(filename)
    data = readlines(filename)
    comments = data[1]
    idata = 1
    idata = get_position("CRYSTAL", data)
    #@assert data[idata] == "CRYSTAL" "Only CRYSTAL format is supported. Now $(data[3])"
    idata = get_position("PRIMVEC", data)
    #@assert data[idata] == "PRIMVEC" "Only PRIMVEC format is supported. Now $(data[4])"
    cell = zeros(3, 3)
    for k = 1:3
        cell[k, :] = parse.(Float64, split(data[idata+k]))
    end
    idata = get_position("PRIMCOORD", data)
    #@assert data[idata] == "PRIMCOORD" "Only PRIMCORD format is supported. Now $(data[idata])"
    #idata = 9
    idata += 1
    numatoms = parse(Int64, split(data[idata])[1])
    R = zeros(3, numatoms)
    F = zeros(3, numatoms)
    kinds = Vector{String}(undef, numatoms)
    species_set = Set()
    for i = 1:numatoms
        idata = idata + 1
        u = split(data[idata])
        #println(u)
        R[:, i] = parse.(Float64, u[2:4])
        F[:, i] = parse.(Float64, u[5:end])
        kinds[i] = u[1]
        push!(species_set, u[1])
    end
    #haslocalinfo = zeros(Bool, numatoms)
    #localinfo = Array{Union{Nothing,LocalRdata}}(undef, numatoms)
    #haslocalRvectors = zeros(Bool, numatoms)
    #localRvectors = Array{Union{Nothing,Matrix{Float64}}}(undef, numatoms)
    species = collect(species_set)
    return XSFdata(R, F, comments, filename, cell, kinds, species, numatoms)#, haslocalinfo, localinfo, haslocalRvectors, localRvectors)
end

function cross_product!(a, b, c)
    c[1] = a[2] * b[3] - a[3] * b[2]
    c[2] = a[3] * b[1] - a[1] * b[3]
    c[3] = a[1] * b[2] - a[2] * b[1]
end


function calculate_pbcbox(cell, Rmax)
    lattice_vector_a = copy(@view cell[1, :])
    lattice_vector_b = copy(@view cell[2, :])
    lattice_vector_c = copy(@view cell[3, :])
    axb = zero(lattice_vector_a)
    bxc = zero(lattice_vector_a)
    axc = zero(lattice_vector_a)

    cross_product!(lattice_vector_a, lattice_vector_b, axb)
    axb ./= norm(axb)
    cross_product!(lattice_vector_a, lattice_vector_c, axc)
    axc ./= norm(axc)
    cross_product!(lattice_vector_b, lattice_vector_c, bxc)
    bxc ./= norm(bxc)

    project_to_a = abs(dot(lattice_vector_a, bxc))
    project_to_b = abs(dot(lattice_vector_b, axc))
    project_to_c = abs(dot(lattice_vector_c, axb))

    pbcbox_x = 0
    pbcbox_y = 0
    pbcbox_z = 0

    while pbcbox_x * project_to_a <= Rmax
        pbcbox_x += 1
    end

    while pbcbox_y * project_to_b <= Rmax
        pbcbox_y += 1
    end

    while pbcbox_z * project_to_c <= Rmax
        pbcbox_z += 1
    end

    return pbcbox_x, pbcbox_y, pbcbox_z

end

function calculate_pbcbox(xsf::XSFdata, Rmax)
    lattice_vector_a = copy(@view xsf.cell[1, :])
    lattice_vector_b = copy(@view xsf.cell[2, :])
    lattice_vector_c = copy(@view xsf.cell[3, :])
    axb = zero(lattice_vector_a)
    bxc = zero(lattice_vector_a)
    axc = zero(lattice_vector_a)

    cross_product!(lattice_vector_a, lattice_vector_b, axb)
    axb ./= norm(axb)
    cross_product!(lattice_vector_a, lattice_vector_c, axc)
    axc ./= norm(axc)
    cross_product!(lattice_vector_b, lattice_vector_c, bxc)
    bxc ./= norm(bxc)

    project_to_a = abs(dot(lattice_vector_a, bxc))
    project_to_b = abs(dot(lattice_vector_b, axc))
    project_to_c = abs(dot(lattice_vector_c, axb))

    pbcbox_x = 0
    pbcbox_y = 0
    pbcbox_z = 0

    while pbcbox_x * project_to_a <= Rmax
        pbcbox_x += 1
    end

    while pbcbox_y * project_to_b <= Rmax
        pbcbox_y += 1
    end

    while pbcbox_z * project_to_c <= Rmax
        pbcbox_z += 1
    end

    return pbcbox_x, pbcbox_y, pbcbox_z

end

function make_Rmatrix(R_js::Vector{Vector{T}}, natoms) where {T<:Real}
    @assert length(R_js) <= natoms "length(R_js) should be smaller than natoms"
    R_j = zeros(T, 3, natoms)
    for i = 1:length(R_js)
        R_j[:, i] .= R_js[i]
    end
    return R_j
end

function make_Rmatrix(R_js::Vector{Vector{T}}, atomkinds_j, natoms, species) where {T<:Real}
    @assert length(R_js) <= natoms "length(R_js) should be smaller than natoms"
    nums = length(species)
    R_j = zeros(T, 4, natoms)
    for i = 1:length(R_js)
        for k = 1:3
            R_j[k, i] = R_js[i][k]
        end
        pos = findfirst(x -> x == atomkinds_j[i], species)
        #println("$i $pos $(atomkinds_j[i]) $species")
        R_j[4, i] = ifelse(nums % 2 == 0, pos - (nums / 2 + 0.5), pos - (nums + 1) / 2)
        #R_j[4,i] = ifelse(atomkinds_j[i]=="H",1,-1)
    end
    return R_j
end

function get_localRvectors(xsf::XSFdata, ith_atom, Rmax, natoms, haskinds=false)
    #if xsf.haslocalRvectors[ith_atom]
    #else
    get_atoms_inside_the_sphere(xsf, ith_atom, Rmax)
    localinfo = xsf.localinfo[ith_atom]
    if haskinds
        R_j = make_Rmatrix(localinfo.R_js, localinfo.atomkinds_j, natoms, xsf.species)
    else
        R_j = make_Rmatrix(localinfo.R_js, natoms)
    end
    #xsf.localRvectors[ith_atom] = deepcopy(R_j)
    #xsf.haslocalRvectors[ith_atom] = true
    #end
    return R_j

    #return xsf.localRvectors[ith_atom]
end

function get_atoms_inside_the_sphere!(numatoms, ith_atom, Rmax, R, kinds, cell, R_i, R_js,
    indices_j, atomkinds_j)
    for p = 1:3
        R_i[p] = R[p, ith_atom]
    end
    a = kinds[ith_atom]
    atomkind_i = a
    index_i = ith_atom
    Rmax2 = Rmax^2

    pbcbox = calculate_pbcbox(cell, Rmax)

    count = 0
    box_min1 = -pbcbox[1]
    box_max1 = pbcbox[1]
    box_min2 = -pbcbox[2]
    box_max2 = pbcbox[2]
    box_min3 = -pbcbox[3]
    box_max3 = pbcbox[3]

    lattice_vector_a = copy(@view cell[1, :])
    lattice_vector_b = copy(@view cell[2, :])
    lattice_vector_c = copy(@view cell[3, :])
    #atomkinds_j = String[]
    #indices_j = Int64[]

    eps = 1e-4
    R_j = zeros(Float64, 3)
    #diff = Vector{Float64}(undef, 3)


    for j = 1:numatoms
        for box_1 = box_min1:box_max1
            for box_2 = box_min2:box_max2
                for box_3 = box_min3:box_max3
                    if (j == ith_atom && box_1 == 0 && box_2 == 0 && box_3 == 0) == false
                        @inbounds for k = 1:3
                            R_j[k] = R[k, j]
                        end
                        #R_j = xsf.R[:, j]
                        for k = 1:3
                            R_j[k] += -box_1 * lattice_vector_a[k] - box_2 * lattice_vector_b[k] - box_3 * lattice_vector_c[k]
                        end

                        Rij2 = 0.0
                        for k = 1:3
                            Rij2 += (R[k, ith_atom] - R_j[k])^2
                        end

                        if Rij2 < Rmax2 && Rij2 > eps
                            count += 1

                            push!(indices_j, j)
                            push!(atomkinds_j, String(kinds[j]))
                            #diff = similar(R_i)

                            for k = 1:3

                                R_js[k, count] = R_j[k] - R[k, ith_atom]
                            end

                            #diff = Vector{Float64}(undef, 3)
                            #for k in 1:3
                            #    #diff[k] = R_j[k] - R_i[k]
                            #    diff[k] = R_j[k] - R[k, ith_atom]
                            #end
                            #@. diff = R_j - R_i
                            #push!(R_js, diff)             # ← OK (新バッファ)
                            #push!(R_js, R_j - R_i)

                        end

                    end
                end
            end
        end
    end
    return atomkind_i, index_i, count
end


function get_atoms_inside_the_sphere!(numatoms, ith_atom, Rmax, R, kinds, cell, R_i, R_js)
    for p = 1:3
        R_i[p] = R[p, ith_atom]
    end
    a = kinds[ith_atom]
    atomkind_i = a
    index_i = ith_atom
    Rmax2 = Rmax^2

    pbcbox = calculate_pbcbox(cell, Rmax)

    count = 0
    box_min1 = -pbcbox[1]
    box_max1 = pbcbox[1]
    box_min2 = -pbcbox[2]
    box_max2 = pbcbox[2]
    box_min3 = -pbcbox[3]
    box_max3 = pbcbox[3]

    lattice_vector_a = copy(@view cell[1, :])
    lattice_vector_b = copy(@view cell[2, :])
    lattice_vector_c = copy(@view cell[3, :])
    atomkinds_j = String[]
    indices_j = Int64[]

    eps = 1e-4
    R_j = zeros(Float64, 3)
    #diff = Vector{Float64}(undef, 3)


    for j = 1:numatoms
        for box_1 = box_min1:box_max1
            for box_2 = box_min2:box_max2
                for box_3 = box_min3:box_max3
                    if (j == ith_atom && box_1 == 0 && box_2 == 0 && box_3 == 0) == false
                        @inbounds for k = 1:3
                            R_j[k] = R[k, j]
                        end
                        #R_j = xsf.R[:, j]
                        for k = 1:3
                            R_j[k] += -box_1 * lattice_vector_a[k] - box_2 * lattice_vector_b[k] - box_3 * lattice_vector_c[k]
                        end

                        Rij2 = 0.0
                        for k = 1:3
                            Rij2 += (R[k, ith_atom] - R_j[k])^2
                        end

                        if Rij2 < Rmax2 && Rij2 > eps
                            count += 1

                            push!(indices_j, j)
                            push!(atomkinds_j, String(kinds[j]))
                            #diff = similar(R_i)

                            for k = 1:3

                                R_js[k, count] = R_j[k] - R[k, ith_atom]
                            end

                            #diff = Vector{Float64}(undef, 3)
                            #for k in 1:3
                            #    #diff[k] = R_j[k] - R_i[k]
                            #    diff[k] = R_j[k] - R[k, ith_atom]
                            #end
                            #@. diff = R_j - R_i
                            #push!(R_js, diff)             # ← OK (新バッファ)
                            #push!(R_js, R_j - R_i)

                        end

                    end
                end
            end
        end
    end
    return atomkind_i, index_i, atomkinds_j, indices_j, count
end
export get_atoms_inside_the_sphere!


function get_atoms_inside_the_sphere(numatoms, ith_atom, Rmax, R, kinds, cell
)

    #R_i = zeros(Float64, 3)
    #for p = 1:3
    #    R_i[p] = R[p, ith_atom]
    #end
    a = kinds[ith_atom]
    atomkind_i = a
    index_i = ith_atom
    Rmax2 = Rmax^2

    pbcbox = calculate_pbcbox(cell, Rmax)

    count = 0
    box_min1 = -pbcbox[1]
    box_max1 = pbcbox[1]
    box_min2 = -pbcbox[2]
    box_max2 = pbcbox[2]
    box_min3 = -pbcbox[3]
    box_max3 = pbcbox[3]

    lattice_vector_a = copy(@view cell[1, :])
    lattice_vector_b = copy(@view cell[2, :])
    lattice_vector_c = copy(@view cell[3, :])
    R_js = Vector{Float64}[]
    atomkinds_j = String[]
    indices_j = Int64[]

    eps = 1e-4
    R_j = zeros(Float64, 3)

    diff = Vector{Float64}(undef, 3)

    for j = 1:numatoms
        for box_1 = box_min1:box_max1
            for box_2 = box_min2:box_max2
                for box_3 = box_min3:box_max3
                    if (j == ith_atom && box_1 == 0 && box_2 == 0 && box_3 == 0) == false
                        @inbounds for k = 1:3
                            R_j[k] = R[k, j]
                        end
                        #R_j = xsf.R[:, j]
                        for k = 1:3
                            R_j[k] += -box_1 * lattice_vector_a[k] - box_2 * lattice_vector_b[k] - box_3 * lattice_vector_c[k]
                        end

                        Rij2 = 0.0
                        for k = 1:3
                            Rij2 += (R[k, ith_atom] - R_j[k])^2
                        end

                        if Rij2 < Rmax2 && Rij2 > eps
                            push!(indices_j, j)
                            push!(atomkinds_j, String(kinds[j]))
                            #diff = similar(R_i)
                            diff = Vector{Float64}(undef, 3)
                            for k in 1:3
                                #diff[k] = R_j[k] - R_i[k]
                                diff[k] = R_j[k] - R[k, ith_atom]
                            end
                            #@. diff = R_j - R_i
                            push!(R_js, diff)             # ← OK (新バッファ)
                            #push!(R_js, R_j - R_i)
                            count += 1
                        end

                    end
                end
            end
        end
    end

    #localinfo = LocalRdata(R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j)
    return R[:, ith_atom], atomkind_i, index_i, R_js, atomkinds_j, indices_j

end

function get_atoms_inside_the_sphere(xsf::XSFdata, ith_atom, Rmax,
)
    numatoms = xsf.numatoms
    #if xsf.haslocalinfo[ith_atom]
    #return xsf.localinfo.R_i,xsf.localinfo.atomkind_i,xsf.localinfo.index_i,xsf.localinfo.R_js, xsf.localinfo.atomkinds_j, xsf.localinfo.indices_j
    #else
    R_i = zeros(Float64, 3)
    for p = 1:3
        R_i[p] = xsf.R[p, ith_atom]
    end
    a = xsf.kinds[ith_atom]
    atomkind_i = a
    index_i = ith_atom
    Rmax2 = Rmax^2

    pbcbox = calculate_pbcbox(xsf, Rmax)

    count = 0
    box_min1 = -pbcbox[1]
    box_max1 = pbcbox[1]
    box_min2 = -pbcbox[2]
    box_max2 = pbcbox[2]
    box_min3 = -pbcbox[3]
    box_max3 = pbcbox[3]

    lattice_vector_a = copy(@view xsf.cell[1, :])
    lattice_vector_b = copy(@view xsf.cell[2, :])
    lattice_vector_c = copy(@view xsf.cell[3, :])
    R_js = Vector{Float64}[]
    atomkinds_j = String[]
    indices_j = Int64[]

    eps = 1e-4
    R_j = zeros(Float64, 3)

    for j = 1:numatoms
        for box_1 = box_min1:box_max1
            for box_2 = box_min2:box_max2
                for box_3 = box_min3:box_max3
                    if (j == ith_atom && box_1 == 0 && box_2 == 0 && box_3 == 0) == false
                        @inbounds for k = 1:3
                            R_j[k] = xsf.R[k, j]
                        end
                        #R_j = xsf.R[:, j]
                        for k = 1:3
                            R_j[k] += -box_1 * lattice_vector_a[k] - box_2 * lattice_vector_b[k] - box_3 * lattice_vector_c[k]
                        end

                        Rij2 = 0.0
                        for k = 1:3
                            Rij2 += (R_i[k] - R_j[k])^2
                        end

                        if Rij2 < Rmax2 && Rij2 > eps
                            push!(indices_j, j)
                            push!(atomkinds_j, String(xsf.kinds[j]))
                            diff = similar(R_i)
                            @. diff = R_j - R_i
                            push!(R_js, diff)             # ← OK (新バッファ)
                            #push!(R_js, R_j - R_i)
                            count += 1
                        end

                    end
                end
            end
        end
    end

    #localinfo = LocalRdata(R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j)
    return R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j

    #xsf.localinfo[ith_atom] = LocalRdata(R_i, atomkind_i, index_i, R_js, atomkinds_j, indices_j)
    #xsf.haslocalinfo[ith_atom] = true
    #return R_i,atomkind_i,index_i,R_js, atomkinds_j, indices_j
    #end
    #localinfo = xsf.localinfo[ith_atom]

    #return localinfo.R_i, localinfo.atomkind_i, localinfo.index_i, localinfo.R_js, localinfo.atomkinds_j, localinfo.indices_j


end


end