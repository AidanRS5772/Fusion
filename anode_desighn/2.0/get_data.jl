using JSON3

for i in 6:2:362
    file_name = "../anode_data/appratures_$(i).json"
    path = "cathode_geometry/app_$(i).json"
    json_data = JSON3.read(file_name)
    new_json_data = Dict("vertices" => json_data["vertices"], "edges" => json_data["edges"])
    open(path, "w") do io
        JSON3.pretty(io, new_json_data)
    end
end
