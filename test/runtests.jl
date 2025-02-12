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

@testset "BPNET.jl" begin
    # Write your tests here.
    downloadtest()
end
