"""
MarketPulse — Interactive Plotly Dash Dashboard
Visualises credit spread forecasts with uncertainty intervals.
"""

import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import base64


# Load results
def load_results():
    base = Path("data/processed")
    with open(base / "baseline_results.json") as f:
        baseline = json.load(f)
    with open(base / "tft_results.json") as f:
        tft = json.load(f)
    with open(base / "conformal_results.json") as f:
        conformal = json.load(f)
    return baseline, tft, conformal


baseline, tft, conformal = load_results()

# App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="MarketPulse",
    assets_folder="/Users/sherylshunlin/Documents/GitHub/marketpulse/src/dashboard/assets",
)

# Metric cards
def metric_card(title, value, subtitle, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H3(value, className=f"text-{color} mb-0"),
            html.Small(subtitle, className="text-muted"),
        ])
    ], className="mb-3")


# Layout
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("📈 MarketPulse", className="mt-4 mb-0"),
            html.P(
                "UK Credit Spread Forecasting · ARIMA vs Prophet vs TFT vs Conformal Prediction",
                className="text-muted mb-4",
            ),
        ])
    ]),

    # Metric summary row
    dbc.Row([
        dbc.Col(metric_card("ARIMA MAE", "0.0441", "MAPE: 5.04%", "secondary"), md=3),
        dbc.Col(metric_card("Prophet MAE", "0.0538", "MAPE: 6.32%", "secondary"), md=3),
        dbc.Col(metric_card("TFT MAE", "0.0245", "MAPE: 2.89%", "warning"), md=3),
        dbc.Col(metric_card("Conformal MAE", "0.0168", "MAPE: 1.98% | Coverage: 86.7%", "success"), md=3),
    ]),

    # Model selector
    dbc.Row([
        dbc.Col([
            html.Label("Select Models to Display:", className="text-muted mb-2"),
            dcc.Checklist(
                id="model-selector",
                options=[
                    {"label": " ARIMA", "value": "arima"},
                    {"label": " Prophet (with intervals)", "value": "prophet"},
                    {"label": " TFT", "value": "tft"},
                    {"label": " Conformal Prediction (with intervals)", "value": "conformal"},
                ],
                value=["arima", "tft", "conformal"],
                inline=True,
                className="mb-3",
                inputStyle={"margin-right": "5px", "margin-left": "15px"},
            ),
        ])
    ]),

    # Main forecast chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="forecast-chart", style={"height": "500px"}),
        ])
    ]),

    # Model comparison table
    dbc.Row([
        dbc.Col([
            html.H5("Model Comparison", className="mt-4 mb-3"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Model"),
                    html.Th("MAE"),
                    html.Th("RMSE"),
                    html.Th("MAPE"),
                    html.Th("Coverage"),
                    html.Th("vs ARIMA"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("ARIMA(1,1,1)"),
                        html.Td("0.0441"),
                        html.Td("0.0604"),
                        html.Td("5.04%"),
                        html.Td("—"),
                        html.Td("baseline"),
                    ]),
                    html.Tr([
                        html.Td("Prophet"),
                        html.Td("0.0538"),
                        html.Td("0.0649"),
                        html.Td("6.32%"),
                        html.Td("90%"),
                        html.Td("-22% worse", style={"color": "red"}),
                    ]),
                    html.Tr([
                        html.Td("TFT"),
                        html.Td("0.0245"),
                        html.Td("0.0303"),
                        html.Td("2.89%"),
                        html.Td("100%"),
                        html.Td("+44% better", style={"color": "lightgreen"}),
                    ]),
                    html.Tr([
                        html.Td("Conformal (Ridge)"),
                        html.Td("0.0168"),
                        html.Td("0.0208"),
                        html.Td("1.98%"),
                        html.Td("86.7%"),
                        html.Td("+62% better", style={"color": "lightgreen"}),
                    ]),
                ])
            ], bordered=True, hover=True, striped=True),
        ])
    ]),

    # SHAP Explainability Section
    dbc.Row([
        dbc.Col([
            html.H5("Model Explainability — SHAP Analysis", className="mt-4 mb-3"),
            html.P(
                "Which macro factors drive credit spread predictions? "
                "SHAP values show each feature's contribution to the forecast.",
                className="text-muted mb-3",
            ),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(
                    html.Img(
                        src="assets/shap_groups.png",
                        style={"width": "100%", "marginTop": "20px"},
                    ),
                    label="Macro Factor Importance",
                ),
                dbc.Tab(
                    html.Img(
                        src="assets/shap_importance.png",
                        style={"width": "100%", "marginTop": "20px"},
                    ),
                    label="Top 20 Features",
                ),
                dbc.Tab(
                    html.Img(
                        src="assets/shap_summary.png",
                        style={"width": "100%", "marginTop": "20px"},
                    ),
                    label="SHAP Summary (Beeswarm)",
                ),
            ])
        ])
    ]),

    html.Hr(className="mt-4"),
    html.P(
        "Data: FRED API (ICE BofA IG OAS) · Models: statsmodels, Prophet, pytorch-forecasting, MAPIE",
        className="text-muted text-center mb-4",
        style={"fontSize": "0.75rem"},
    ),

], fluid=True)


@app.callback(
    Output("forecast-chart", "figure"),
    Input("model-selector", "value"),
)
def update_chart(selected_models):
    dates = conformal["test_dates"]
    actual = conformal["actual"]

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        name="Actual",
        line=dict(color="white", width=2),
        mode="lines",
    ))

    colors = {
        "arima": "#7B68EE",
        "prophet": "#FFA500",
        "tft": "#FFD700",
        "conformal": "#00FF7F",
    }

    if "arima" in selected_models:
        fig.add_trace(go.Scatter(
            x=dates, y=baseline["arima_forecast"],
            name="ARIMA",
            line=dict(color=colors["arima"], width=1.5, dash="dash"),
        ))

    if "prophet" in selected_models:
        fig.add_trace(go.Scatter(
            x=dates, y=baseline["prophet_forecast"],
            name="Prophet",
            line=dict(color=colors["prophet"], width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=baseline["prophet_upper"] + baseline["prophet_lower"][::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prophet 90% PI",
            showlegend=True,
        ))

    if "tft" in selected_models:
        tft_dates = dates[:len(tft["predictions"])]
        fig.add_trace(go.Scatter(
            x=tft_dates, y=tft["predictions"],
            name="TFT",
            line=dict(color=colors["tft"], width=2),
        ))

    if "conformal" in selected_models:
        fig.add_trace(go.Scatter(
            x=dates, y=conformal["predictions"],
            name="Conformal",
            line=dict(color=colors["conformal"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=conformal["upper"] + conformal["lower"][::-1],
            fill="toself",
            fillcolor="rgba(0,255,127,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Conformal 90% PI",
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_dark",
        title="IG Credit Spread Forecast — Model Comparison",
        xaxis_title="Date",
        yaxis_title="OAS (bps / %)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)