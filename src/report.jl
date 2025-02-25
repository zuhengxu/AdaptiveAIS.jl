all_reports() = [  
        # header with    # lambda expression used to 
        # width of 9     # compute that report item
        "    T     "   => a -> length(a.schedule),
        "    N     "   => a -> n_particles(a.particles),
        "  time(s) "   => a -> a.full_timing.time, 
        "  %t in k "   => a -> percent_time_in_kernel(a),
        "  allc(B) "   => a -> a.timing.bytes,
        "   ess    "   => a -> ess(a.particles),
        "   elbo   "   => a -> a.particles.elbo,
        "    Λ     "   => a -> a.barriers.globalbarrier,
        "log(Z₁/Z₀)"   => a -> a.particles.log_normalization,
    ]

percent_time_in_kernel(a) = a.timing.time / a.full_timing.time

function report(a::AIS_output, is_first, is_last)
    reports = reports_available(a)
    if is_first 
        Pigeons.header(reports) 
    end
    println(
        join(
            map(
                pair -> Pigeons.render_report_cell(pair[2], a),
                reports),
            " "
        ))
    if is_last
        Pigeons.hr(reports, "─")
    end
    return nothing
end
report(a::AIS_output, show_report::Bool) = show_report ? report(a, true, true) : nothing

function reports_available(a::AIS_output)
    result = Pair[] 
    for pair in all_reports() 
        try 
            (pair[2])(a) 
            push!(result, pair)
        catch 
            # some recorder has not been used, skip
        end
    end
    return result
end

process_result(a::AIS_output, method::String) = Dict(
        "method" => method,
        "T" => length(a.schedule),
        "N" => n_particles(a.particles),
        "time(s)" => a.full_timing.time,
        "allc(B)" => a.timing.bytes,
        "ess" => ess(a.particles),
        "elbo" => a.particles.elbo,
        "Λ" => isnothing(a.barriers) ? "NA" : a.barriers.globalbarrier,
        "log(Z₁/Z₀)" => a.particles.log_normalization,
        "Z₁/Z₀" => exp(a.particles.log_normalization),
    )

