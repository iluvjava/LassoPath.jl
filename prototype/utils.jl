function VanderMonde(x::Array{T}, deg::Int64) where {T <: Real}
    """
        Get the vandermond matrix from a column vector, only support 
        real numbers for now. 
    """

    @assert ndims(x) == 2 "expect x to be a column vector of size (n, 1)"*
    string("But its dimension is: ", ndims(x))
    @assert size(x, 2) == 1 "Expect the second dimension to have size 1"*
    string(" but the actual size for x is: ", size(x))

    n = size(x, 1)
    V = zeros(n, deg) # fill with the same type. 
    V[:, 1] = x
    for Col ∈ 2:deg
        V[:, Col] = x.^Col
    end
    return V
end

function PrintTitle(mesg::String)
    """
    
        Print a beautifully formated title to the console. stdout. 
    """
    width = displaysize(stdout)[2]
    println(repeat("=" ,width))
    print("| ")
    println(mesg)
    println(repeat("=" ,width))

end


function Warn(mesg::String, newline=true) 
    printstyled("$mesg $(newline ? '\n' : ' ')", color=:yellow)
end


