###
function nice_print(d::Dict)
    println("-------------------") 
    for (k, v) in d
        println(k, " => ", v)
    end
end


function check_bias(prob, sched; N = 2000, show_report = false)
    method = string(typeof(sched))
    a = ais(prob, sched; N = N, show_report = show_report)
    β = a.schedule
    res_bias = process_result(a, string(typeof(sched)))
    
    FS = FixedSchedule(β)
    a_fs = ais(prob, FS; N = N, compute_barriers = true, show_report = show_report)
    res_fs = process_result(a_fs, "FixedSchedule"*"_"*method)
    
    diffs = Dict(
        "method" => method,
        "T" => length(FS),
        "N" => N,
        "time(s)" => res_bias["time(s)"] - res_fs["time(s)"],
        "allc(B)" => res_bias["allc(B)"] - res_fs["allc(B)"],
        "ess" => res_bias["ess"] - res_fs["ess"],
        "elbo" => res_bias["elbo"] - res_fs["elbo"],
        "log(Z₁/Z₀)" => res_bias["log(Z₁/Z₀)"] - res_fs["log(Z₁/Z₀)"],
        "Z₁/Z₀" => res_bias["Z₁/Z₀"] - res_fs["Z₁/Z₀"],
    ) 
    println("bias for ", method)
    nice_print(diffs)
    return res_bias, res_fs, diffs
end

