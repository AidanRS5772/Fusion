using JSON3

function clean_data(app_cnt)
    file_name = "cathode_data/appratures_$(app_cnt).json"
    original_data = JSON3.read(file_name)

    cleaned_data = Dict(
        "edges" => original_data["edges"],
        "vertices" => original_data["vertices"],
        "points" => original_data["points"]  # Note: matching "points" to "face_centers"
    )

    # Write the cleaned data to a new file
    open(file_name, "w") do io
        JSON3.write(io, cleaned_data)
    end
end

for i in 6:2:362
    clean_data(i)
end
