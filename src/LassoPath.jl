module LassoPath
    
       using COSMO, JuMP, LinearAlgebra
       using Statistics
       import Plots as Plots
       import ProgressMeter as Pm
       MOI = JuMP.MathOptInterface

       # Write your package code here.
       
       # The order here matters
       include("utils.jl")
       include("model_build.jl")
       include("lasso_scop.jl")
       include("lasso_methods.jl")

       export vanderMonde, 
              PrintTitle, 
              Warn

       export LassoSCOP,
           GetLassoPath,
           VisualizeLassoPath, 
           CaptureImportantWeights
    

end
