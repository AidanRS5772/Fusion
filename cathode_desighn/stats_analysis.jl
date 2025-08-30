using PlotlyJS
using JSON3

function parse_json(file_name)
	json_str = read(file_name, String)
	return JSON3.read(json_str, Dict)
end

function get_data_app_cnt(app_cnts, file_name, path)
	json_dict = parse_json(file_name)
	props = split(path, "/")
	data = []
	for app_cnt in app_cnts
		app_cnt_data = json_dict["app_cnt_$app_cnt"]
		for prop in props	
			app_cnt_data = app_cnt_data[prop]
		end
		push!(data, app_cnt_data);
	end

	return data
end

file_name = "MC_data.json"
app_cnts = 6:2:170;
med = get_data_app_cnt(app_cnts, file_name, "fusion/percentiles/median")

plot(scatter(
	x = collect(app_cnts),
	y = data,
	mode = "lines+markers",
	name = "Fusion Probability Median"
))


