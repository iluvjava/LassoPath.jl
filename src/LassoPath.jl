module LassoPath
    
    # Write your package code here.
    
    include("lasso_methods.jl")
    include("utils.jl")
    include("lasso_scop.jl")
    include("model_build.jl")

    export vanderMonde, 
           PrintTitle, 
           Warn

    export LassoSCOP,
           GetLassoPath, 
           VisualizeLassoPath, 
           CaptureImportantWeights
    

end
