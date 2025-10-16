import json
import plotly.graph_objects as go
import numpy as np


def get_data(app_cnts, json_data, path):
    keys = path.split("/")
    data = []
    for app_cnt in app_cnts:
        app_data = json_data[f"app_cnt_{app_cnt}"]
        for key in keys:
            app_data = app_data[key]
        data.append(app_data)
    return data


def main():
    with open("../MC_data.json", "r") as file:
        json_data = json.load(file)

    app_cnts = list(range(6, 170, 2))
    app_cnts.remove(160);
    xi = get_data(app_cnts, json_data, "distributions/pareto/xi")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = app_cnts,
            y = xi,
            mode = "lines+markers",
        )
    )

    fig.update_layout(
        title="Generalized Pareto XI values",
        xaxis_title="App Counts",
    )

    fig.show()


if __name__ == "__main__":
    main()
