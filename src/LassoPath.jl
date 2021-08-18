module LassoPath
    # Write your package code here.
    include("lasso.jl")
    include("utils.jl")

    export vanderMonde, 
           PrintTitle, 
           Warn
    export LassoSCOP,
           GetLassopath, 
           VisualizeLassoPath, 
           CaptureImportantWeights
    

end
