using PosteriorDB, StanLogDensityProblems, BridgeStan

function get_stan_model(model_name::String)
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, model_name)
    StanProblem(post, ".stan/"; force = true)
end







