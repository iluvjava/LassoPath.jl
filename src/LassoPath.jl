module LassoPath
    
       using COSMO, JuMP, LinearAlgebra
       using Statistics
       import Plots as Plots
       import ProgressMeter as Pm
       MOI = JuMP.MathOptInterface

       # Write your package code here.
       
       # The order here matters
       include("utils.jl")
       include("lasso_root.jl")
       include("model_make.jl")
       include("lasso_scop.jl")
       include("prox.jl")
       include("lasso_prox.jl")
       include("lasso_methods.jl")

       export vanderMonde, 
              PrintTitle, 
              Warn

       export LassoRoot, 
              LassoSCOP,
              Lasso2NormProx,
              GetLassoPath,
              VisualizeLassoPath, 
              CaptureImportantWeights

       
       # For testing internal metods
       # export BuildProximalGradientLasso,
       #        OptimizeProximalGradient
    

end
