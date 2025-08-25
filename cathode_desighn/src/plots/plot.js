const fs = require("fs");
const path = require("path");
const open = require("open");

const type = process.argv[2];
const projectRoot = process.argv[3]; // Pass PROJECT_ROOT as 3rd argument
const hash = process.argv[4]; // Hash becomes 4th argument when needed

// Define all paths relative to project root
const PATHS = {
	meshMetaData: path.join(projectRoot, "mesh_meta_data.json"),
	plotData: path.join(projectRoot, "src/plots/plot_data.json"),
	outputHtml: path.join(projectRoot, "src/plots/plot.html"),
};

const traces = [];
let layout = {};

if (type === "0") {
	const all_mesh_data = JSON.parse(
		fs.readFileSync(PATHS.meshMetaData, "utf-8"),
	);
	const mesh_data = all_mesh_data[hash];

	const mesh_trace = {
		type: "mesh3d",
		x: mesh_data.X,
		y: mesh_data.Y,
		z: mesh_data.Z,
		i: mesh_data.I,
		j: mesh_data.J,
		k: mesh_data.K,
		opacity: 0.5,
		color: "grey",
		hoverinfo: "skip",
	};

	traces.push(mesh_trace);
	layout = {
		title: `${mesh_data.app_cnt} Aperture Mesh`,
		scene: {
			xaxis: {
				title: "X (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			yaxis: {
				title: "Y (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			zaxis: {
				title: "Z (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			aspectmode: "cube",
		},
	};
} else if (type === "1") {
	const all_mesh_data = JSON.parse(
		fs.readFileSync(PATHS.meshMetaData, "utf-8"),
	);
	const mesh_data = all_mesh_data[hash];

	const mesh_trace = {
		type: "mesh3d",
		x: mesh_data.X,
		y: mesh_data.Y,
		z: mesh_data.Z,
		i: mesh_data.I,
		j: mesh_data.J,
		k: mesh_data.K,
		opacity: 0.5,
		color: "grey",
		hoverinfo: "skip",
	};

	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const path_trace = {
		type: "scatter3d",
		x: data.X,
		y: data.Y,
		z: data.Z,
		mode: "lines",
		line: {
			color: data.S,
			colorscale: "Viridis",
			width: 3,
		},
		hovertemplate: `X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Speed: %{line.color:.2f} cm/ns <extra></extra>`,
	};
	traces.push(mesh_trace, path_trace);
	layout = {
		title: `${mesh_data.app_cnt} Aperture Mesh and Trajectory`,
		scene: {
			xaxis: {
				title: "X (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			yaxis: {
				title: "Y (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			zaxis: {
				title: "Z (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			aspectmode: "cube",
		},
	};
} else if (type === "2") {
	const all_mesh_data = JSON.parse(
		fs.readFileSync(PATHS.meshMetaData, "utf-8"),
	);
	const mesh_data = all_mesh_data[hash];

	const mesh_trace = {
		type: "mesh3d",
		x: mesh_data.X,
		y: mesh_data.Y,
		z: mesh_data.Z,
		i: mesh_data.I,
		j: mesh_data.J,
		k: mesh_data.K,
		opacity: 0.5,
		color: "grey",
		hoverinfo: "skip",
	};

	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const cone_trace = {
		type: "cone",
		x: data.X,
		y: data.Y,
		z: data.Z,
		u: data.U,
		v: data.V,
		w: data.W,
		colorscale: "Viridis",
		cmin: Math.min(...data.M),
		cmax: Math.max(...data.M),
		colorbar: {
			title: "Electric Field Magnitude",
		},
		showscale: true,
		sizemode: "absolute",
		sizeref: 0.1,
		anchor: "tail",
		customdata: data.M,
		hovertemplate:
			"X: %{x}<br>Y: %{y}<br>Z: %{z}<br>" +
			"Electric Field Magnitude: %{customdata:.2f}<extra></extra>",
	};

	traces.push(mesh_trace, cone_trace);
	layout = {
		title: `${mesh_data.app_cnt} Aperture Mesh and Electric Field`,
		scene: {
			xaxis: {
				title: "X (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			yaxis: {
				title: "Y (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			zaxis: {
				title: "Z (cm)",
				range: [-mesh_data.anode_radius, mesh_data.anode_radius],
			},
			aspectmode: "cube",
		},
	};
} else if (type === "3") {
	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const ke_trace = {
		type: "scatter",
		x: data.T,
		y: data.KE,
		mode: "lines",
		name: "Kinetic",
		line: {
			color: "red",
		},
	};
	const pe_trace = {
		type: "scatter",
		x: data.T,
		y: data.PE,
		mode: "lines",
		name: "Potential",
		line: {
			color: "blue",
		},
	};
	const tot_trace = {
		type: "scatter",
		x: data.T,
		y: data.E,
		mode: "lines",
		name: "Total Energy",
		line: {
			color: "grey",
		},
	};

	traces.push(ke_trace, pe_trace, tot_trace);
	layout = {
		title: "Energy Plot",
		showlegend: true,
	};
} else if (type === "4") {
	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const maxVal = Math.max(...data.FP);
	const hist = {
		type: "histogram",
		x: data.FP,
		histnorm: "probability density",
		autobinx: false,
		xbins: {
			start: 0.0,
			end: maxVal,
			size: 1,
		},
		marker: {
			color: "blue",
		},
		opacity: 0.6,
		hovertemplate:
			"<b>Fusion Probability:</b> %{x}<br>" +
			"<b>Frequency:</b> %{y}<br>" +
			"<extra></extra>",
	};

	traces.push(hist);

	layout = {
		title: "Fusion Probability Distribution",
		showlegend: true,
	};
} else if (type === "5") {
	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const maxVal = Math.max(...data.FP);
	const hist = {
		type: "histogram",
		x: data.FP,
		histnorm: "probability density",
		autobinx: false,
		xbins: {
			start: 0.0,
			end: maxVal,
			size: maxVal / 100,
		},
		marker: {
			color: "blue",
		},
		opacity: 0.6,
		hovertemplate:
			"<b>Fusion Probability:</b> %{x}<br>" +
			"<b>Frequency:</b> %{y}<br>" +
			"<extra></extra>",
	};

	const exp_fit = {
		type: "scatter",
		x: data.X,
		y: data.EXP_PDF,
		mode: "lines",
		name: "Exponential Fit",
	}
	const pareto_fit = {
		type: "scatter",
		x: data.X,
		y: data.PARETO_PDF,
		mode: "lines",
		name: "Generalized Pareto Fit",
	}
	const hazard_fit = {
		type: "scatter",
		x: data.X,
		y: data.HAZARD_PDF,
		mode: "lines",
		name: "Hazard Fit",
	}


	traces.push(hist, exp_fit, pareto_fit, hazard_fit);

	layout = {
		title: "Fusion Probability Distribution",
		showlegend: true,
	};
} else if (type === "6") {
	const data = JSON.parse(fs.readFileSync(PATHS.plotData, "utf-8"));

	const cdf = {
		type: "scatter",
		x: data.FP_CDF_X,
		y: data.FP_CDF_Y,
		mode: "markers",
		marker: {
			size: 1
		},
		name: "Emperical CDF"
	}

	const exp_fit = {
		type: "scatter",
		x: data.X,
		y: data.EXP_PDF,
		mode: "lines",
		name: "Exponential Fit",
	}
	const pareto_fit = {
		type: "scatter",
		x: data.X,
		y: data.PARETO_PDF,
		mode: "lines",
		name: "Generalized Pareto Fit",
	}
	const hazard_fit = {
		type: "scatter",
		x: data.X,
		y: data.HAZARD_PDF,
		mode: "lines",
		name: "Hazard Fit",
	}


	traces.push(cdf, exp_fit, pareto_fit, hazard_fit);

	layout = {
		title: "Fusion Cumulitive Density Function",
		showlegend: true,
	};
}



const html = `
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Plot</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
  <body>
    <div id="plot" style="width:100%; height:100vh;"></div>
    <script>
      const data = ${JSON.stringify(traces)};
      const layout = ${JSON.stringify(layout)};
      Plotly.newPlot("plot", data, layout);
    </script>
  </body>
</html>
`;

fs.writeFileSync(PATHS.outputHtml, html);
open(PATHS.outputHtml);
