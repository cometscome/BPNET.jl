using BPNET
using Test

using Downloads
using CodecBzip2
using Tar
using TranscodingStreams



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
    datafiles = readdir(exampledir, join=true)
    adddata!(g, datafiles)
    set_numfiles!(g, 1000)


    display(g.numfiles)

end

@testset "BPNET.jl" begin
    # Write your tests here.
    generatetest()
    #downloadtest()
end
