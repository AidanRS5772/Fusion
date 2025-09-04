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
    median = np.log(get_data(app_cnts, json_data, "fusion/percentiles/median"))
    p75 = np.log(get_data(app_cnts, json_data, "fusion/percentiles/75"))
    p90 = np.log(get_data(app_cnts, json_data, "fusion/percentiles/90"))
    p95 = np.log(get_data(app_cnts, json_data, "fusion/percentiles/95"))
    p99 = np.log(get_data(app_cnts, json_data, "fusion/percentiles/95"))
    max = np.log(get_data(app_cnts, json_data, "fusion/fusion_max"))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=median,
            mode="lines+markers",
            name="Median",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=p75,
            mode="lines+markers",
            name="75%",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=p90,
            mode="lines+markers",
            name="90%",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=p95,
            mode="lines+markers",
            name="95%",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=p99,
            mode="lines+markers",
            name="99%",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=list(app_cnts),
            y=max,
            mode="lines+markers",
            name="Maximum",
        )
    )

    fig.show()


if __name__ == "__main__":
    main()
