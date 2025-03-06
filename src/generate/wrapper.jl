import BPNET_jll

function generate(filename)
    exe = BPNET_jll.generate()
    run(`$exe $filename`)
    #BPNET_jll.generate() do exe
    #    run(`$exe $filename`)
    #end
end
export generate