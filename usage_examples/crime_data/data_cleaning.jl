
cd("C:\\Users\\victo\\source\\repos\\Silly_Julia_stuff\\Julia Packages\\LassoPath\\usage_examples\\crime_data")
using DelimitedFiles, CSV
using DataFrames

D = readdlm("communities.csv", ',')
Header = readdlm("header_info.txt", ' ')
Df = DataFrame(D, :auto)
rename!(Df, Symbol.(reshape(Header[:, 1], :)))

# convert the types
function FilterTheDataFrame(Df::DataFrame)
    GoodColumns = []
    for (II, Col) in enumerate(eachcol(Df))
        if !("?" ∈ Df[:, II])
            push!(GoodColumns, II)
        end
        convertedType = Header[II, 2] == "numeric" ? Float64 : String
        convertFunc = entry -> begin
            if entry == "?"
                return NaN64
            else
                return convert(convertedType, entry)
            end
        
        end
        Df[!, II] = convertFunc.(Df[:, II])  # To include type, `!` will be needed. 
    end
    return Df[!, GoodColumns]
end

Df = FilterTheDataFrame(Df)
Df = Df[!, Not(2)]
describe(Df)

# Save the processed DataFrame
CSV.write("process_data.csv", Df)

