# --- Imports ---
import ast
import re
import codecs
import io
# import asyncio # Not needed for this approach
import json
import time
import dash_auth
import traceback
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import pandas as pd
import requests
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import helpers
import os
import Dashauth
import flight_client
import schedule
import threading

# --- Constants ---
API_BASE_URL = "http://13.239.238.138:5000" # API endpoint being polled
FETCH_LIMIT = 200
FETCH_INTERVAL_SECONDS = 5
PERSISTENCE_HOURS = 24
MAX_STORE_RECORDS = 5000000
POST_API_URL = "https://ss-data-services.onrender.com/api/flights"
MAP_CENTER = {"lat": 7.0, "lon": 30.0}
MAP_ZOOM = 4.5
MAP_LINE_WIDTH = 0.5
MAP_MARKER_OPACITY = 0.8
MAP_POINT_SIZE_MAX = 8
FLIGHT_ID_COLUMN = "flight_id"
MAPBOX_STYLE = "carto-darkmatter"
AIRCRAFT_SYMBOL = 'circle'
ON_MAP_TEXT_COLOR = 'white' if MAPBOX_STYLE in ["carto-darkmatter", "dark", "satellite", "streets-dark"] else 'black'
ON_MAP_TEXT_SIZE = 9
ON_MAP_TEXT_POSITION = 'middle center'
DISPLAY_TIMEZONE = pytz.timezone('Africa/Nairobi')
PROCESSING_TIMEZONE = pytz.utc
APP_START_TIME = datetime.now()
PLOTLY_TEMPLATE = "plotly_dark"
CHART_FONT_COLOR = "#adb5bd"
CHART_PAPER_BG = 'rgba(0,0,0,0)'
CHART_PLOT_BG = 'rgba(0,0,0,0)'
NA_REPLACE_VALUES = ['', 'nan', 'NaN', 'None', 'null', 'NONE', 'NULL', '#N/A', 'N/A', 'NA', '-']
CSV_FILE_PATH = "flight_data.csv"

# --- Initialize Fetcher ---
fetcher = None
if flight_client:
    try:
        fetcher = flight_client.FlightDataFetcher(base_url=API_BASE_URL, api_endpoint="/api/flights/recent", fetch_limit=FETCH_LIMIT)
        print(f"FlightDataFetcher initialized for API: {API_BASE_URL}")
    except Exception as e:
        print(f"ERROR: Failed to initialize FlightDataFetcher: {e}")
        fetcher = None
else:
    print("ERROR: flight_client module not loaded, cannot initialize fetcher.")

# --- Initialize App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.VAPOR],
    suppress_callback_exceptions=True, title="SS RT Analytics"
)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
server = app.server

auth = dash_auth.BasicAuth(
    app,
    Dashauth.VALID_USERNAME_PASSWORD_PAIRS
)

# --- Helper Functions ---
def create_empty_map_figure(message="No data available"):
    map_font_color = ON_MAP_TEXT_COLOR
    layout = go.Layout(
        mapbox=dict(style=MAPBOX_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
        margin=dict(r=5, t=5, l=5, b=5), paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PLOT_BG, font=dict(color=map_font_color) )
    layout.annotations = [go.layout.Annotation( text=message, align='center', showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=16, color=CHART_FONT_COLOR) )]
    return go.Figure(data=[], layout=layout)

def create_empty_analytics_figure(title="No data"):
    fig = go.Figure()
    fig.update_layout( template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10), annotations=[dict( text=title, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color=CHART_FONT_COLOR) )])
    return fig

def load_data_from_csv():
    """Loads data from the CSV file if it exists."""
    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            return df
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def delete_csv():
    """Deletes the CSV file."""
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)
        print(f"CSV file '{CSV_FILE_PATH}' deleted.")

def schedule_deletion():
    """Schedules the CSV file deletion every Sunday at midnight based on the system's timezone."""
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    schedule.every().sunday.at("00:00").do(delete_csv)
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("CSV deletion scheduled every Sunday at 00:00 (system timezone).")

# --- Main Application Layout ---
app.layout = dbc.Container([

    dcc.Download(id="download-csv"),

    # Header
    dbc.Row( [ html.Br(), dbc.Col( html.Img(src=app.get_asset_url("ss_logo.png"), height="40px", className="rounded-circle"), className="mt-3" ) ], align="center", className="g-0", ),
    dbc.Row([ dbc.Col(html.H1("South Sudan Flight Tracker", className="text-center text-primary mb-2"), width=12), dbc.Col(html.P(id='last-update-timestamp', className="text-center text-muted mb-4 small"), width=12) ]),

    # Map and Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Map Controls"),
                dbc.CardBody([
                    html.Div([html.Label("Display Mode:", className="fw-bold"), dbc.RadioItems(id="map-display-mode", options=[{"label": "Routes", "value": "routes"},{"label": "Current", "value": "current"}], value="current", inline=True, inputClassName="ms-1 me-2")], className="mb-3"),
                    html.Div([html.Label("Aircraft Filter:", className="fw-bold"), dcc.Dropdown(id="map-aircraft-filter", multi=True, placeholder="Select aircraft types...") ], style={"color":"black"}, className="mb-3"),
                    html.Div([html.Label("Altitude Range (ft):", className="fw-bold"), dcc.RangeSlider(id="map-altitude-range", min=0, max=50000, step=1000, marks={i * 10000: {'label': f"{i*10}k"} for i in range(6)}, value=[0, 50000], tooltip={"placement": "bottom", "always_visible": False}, className="p-0")]),
                    html.Div(dbc.Button("Model Insights", id="insights-button", color="success", className="w-100 mt-4"), className="d-grid gap-2")
                ])
            ], className="mb-4")
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Flight Map"),
                dbc.CardBody(dcc.Graph(id="flight-map", style={"height": "65vh"}, config={"displayModeBar": True, "scrollZoom": True}), className="p-0")
            ], className="mb-4")
        ], width=12, lg=9)
    ], className="mb-4"),

    # Analytics Rows (Keep structure, data source needs checking)
    dbc.Row([ dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Active Flights"), id="header-total-flights"), dbc.CardBody(html.H4(id="analytics-total-flights", className="text-primary text-center my-auto", children="-"))], className="h-100 d-flex flex-column"), width=12, lg=4, className="mb-3"), dbc.Tooltip("Total unique flights visible based on current filters & latest data.", target="header-total-flights", placement='bottom'), dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Avg. Speed (kts)"), id="header-avg-speed"), dbc.CardBody(html.H4(id="analytics-avg-speed", className="text-primary text-center my-auto", children="-"))], className="h-100 d-flex flex-column"), width=12, lg=4, className="mb-3"), dbc.Tooltip("Avg ground speed (kts) of visible flights (latest position).", target="header-avg-speed", placement='bottom'), dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Flights Per Day"), id="header-flights-day"), dbc.CardBody(dcc.Graph(id='analytics-flights-per-day', figure={}, style={'height': '150px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=4, className="mb-3"), dbc.Tooltip(f"Unique flights per day (based on data within the last {PERSISTENCE_HOURS} hours).", target="header-flights-day", placement='bottom'), ], className="mb-2 g-3"),
    dbc.Row([ dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Top Aircraft Types"), id="header-aircraft-types"), dbc.CardBody(dcc.Graph(id='analytics-aircraft-types', figure={}, style={'height': '200px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=4, className="mb-3"), dbc.Tooltip("Top 10 aircraft types (latest report, visible flights).", target="header-aircraft-types", placement='bottom'), dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Altitude Distribution"), id="header-altitude-dist"), dbc.CardBody(dcc.Graph(id='analytics-altitude-dist', figure={}, style={'height': '200px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=4, className="mb-3"), dbc.Tooltip("Altitude distribution (ft) (latest report, visible flights).", target="header-altitude-dist", placement='bottom'), dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Speed Distribution"), id="header-speed-dist"), dbc.CardBody(dcc.Graph(id='analytics-speed-dist', figure={}, style={'height': '200px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=4, className="mb-3"), dbc.Tooltip("Ground speed distribution (kts) (latest report, visible flights).", target="header-speed-dist", placement='bottom'), ], className="mb-2 g-3"),
    dbc.Row([ dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Top 5 Origins"), id="header-top-origins"), dbc.CardBody(dcc.Graph(id='analytics-top-origins', figure={}, style={'height': '200px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=6, className="mb-3"), dbc.Tooltip("Top 5 origin airports (excluding N/A) (latest report, visible flights).", target="header-top-origins", placement='bottom'), dbc.Col(dbc.Card([dbc.CardHeader(html.H6("Top 5 Routes"), id="header-top-routes"), dbc.CardBody(dcc.Graph(id='analytics-top-routes', figure={}, style={'height': '200px'}, config={'displayModeBar': False}), className="p-1")], className="h-100"), width=12, lg=6, className="mb-3"), dbc.Tooltip("Top 5 routes (Origin -> Dest, excluding N/A) (latest report, visible flights).", target="header-top-routes", placement='bottom'), ], className="mb-2 g-3"),

    # Data Table Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Flight Data (Latest Updates Per Flight in Period)"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="flight-table",
                        style_table={"overflowX": "auto"},
                        style_header={'backgroundColor': '#eee', 'color': '#333', 'fontWeight': 'bold', 'border': '1px solid #ddd'},
                        style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': 'white', 'color': '#333', 'border': '1px solid #eee'},
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                            {'if': {'state': 'selected'}, 'backgroundColor': 'rgba(0, 116, 217, 0.3)', 'border': '1px solid #0074D9'}
                        ],
                        page_size=15, sort_action='native', filter_action='native',
                        row_selectable='multi', selected_rows=[], row_deletable=False,
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button("Post Selected", id="post-button", color="info", className="mt-3 me-2", n_clicks=0), width="auto"),
                        dbc.Col(dbc.Button("Download CSV", id="download-button", color="secondary", className="mt-3", n_clicks=0), width="auto")
                    ], justify="start", className="mt-2"),
                    html.Div(id="post-feedback", className="mt-3 text-muted small")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    # Modal
    dbc.Modal([ dbc.ModalHeader(dbc.ModalTitle("Flight Data Insights")), dcc.Loading(id="loading-insights1", children=[dbc.ModalBody(id="insights-modal-body")]), dbc.ModalFooter(dcc.Loading(dbc.Button("Close", id="insights-modal-close", className="ms-auto", n_clicks=0), id="loading-insights", type="cube")), ], id="insights-modal", is_open=False, size="xl", backdrop=True, scrollable=True),

    # Interval Component
    dcc.Interval(id="interval-component", interval=FETCH_INTERVAL_SECONDS * 1000, n_intervals=0),

    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("©️ 2025 South Sudan Civil Aviation. Powered by Crawford Capital", className="text-center text-muted mt-4"),
                html.Div([
                    dbc.Button("Report Issue", color="link", className="text-muted", href="#", id="report-issue-btn"),
                    html.Span(" | "),
                    dbc.Button("Privacy Policy", color="link", className="text-muted", id="privacy-policy-btn", n_clicks=0)
                ], className="text-center small")
            ], className="py-4")
        ])
    ])

], fluid=True, className="ag-theme-alpine-dark")

# --- Start the scheduled deletion ---
schedule_deletion()

# --- Callbacks ---

@callback(
    Output('last-update-timestamp', 'children'),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def update_data(n_intervals):
    """
    Callback triggered every FETCH_INTERVAL_SECONDS.
    Fetches the latest batch of new flight data from the API and stores it in a CSV.
    """
    global fetcher

    if fetcher is None:
        timestamp = datetime.now(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        print("ERROR in update_data: Fetcher not available.")
        return f"Fetcher Init Error at {timestamp}"

    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Interval {n_intervals}: update_data triggered.")
    fetch_time = datetime.now(DISPLAY_TIMEZONE)
    error_msg_base = f"Data Update Failed (Interval {n_intervals} at {fetch_time.strftime('%H:%M:%S %Z')})"
    success_msg_base = f"Data Update Check OK (Interval {n_intervals} at {fetch_time.strftime('%H:%M:%S %Z')})"
    status_msg = f"Last check: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"

    try:
        df_batch = fetcher.fetch_next_batch()

        if df_batch is None:
            print(f"ERROR in update_data: fetch_next_batch() returned None (API fetch error).")
            return f"{error_msg_base} (API Fetch Error)"

        if df_batch.empty:
            print(f"No new flight data received since last check.")
            last_ts_msg = fetcher.last_processed_timestamp or "initial fetch"
            return f"{success_msg_base} (No New Data since {last_ts_msg})"

        print(f"Fetched batch with {len(df_batch)} new/updated records.")

        df_processed_batch = df_batch.copy()
        df_processed_batch.columns = df_processed_batch.columns.str.lower().str.strip()

        col_map = {
            'lat': 'LATITUDE', 'lon': 'LONGITUDE', 'last_update': 'LAST_UPDATE_TIME',
            'flight_id': FLIGHT_ID_COLUMN, 'model': 'AIRCRAFT_MODEL', 'alt': 'ALTITUDE',
            'speed': 'SPEED', 'track': 'TRACK', 'callsign': 'FLIGHT_CALLSIGN',
            'reg': 'REGISTRATION', 'origin': 'ORIGIN', 'destination': 'DESTINATION',
            'flight': 'FLIGHT_NUMBER'
        }
        rename_map = {k: v for k, v in col_map.items() if k in df_processed_batch.columns}
        if rename_map:
            df_processed_batch.rename(columns=rename_map, inplace=True)

        essential_cols = ['LATITUDE', 'LONGITUDE', 'LAST_UPDATE_TIME', FLIGHT_ID_COLUMN]
        if not all(col in df_processed_batch.columns for col in essential_cols):
            missing = [c for c in essential_cols if c not in df_processed_batch.columns]
            print(f"Warning: Batch missing essential columns after rename: {missing}. Skipping batch.")
            return f"{error_msg_base} (Missing Essential Cols)"

        if "LAST_UPDATE_TIME" in df_processed_batch.columns:
            df_processed_batch["LAST_UPDATE_TIME"] = pd.to_datetime(
                df_processed_batch["LAST_UPDATE_TIME"], errors='coerce', utc=True)
            df_processed_batch["LAST_UPDATE_TIME"] = df_processed_batch["LAST_UPDATE_TIME"].fillna(pd.NaT)
        else:
            print("Warning: LAST_UPDATE_TIME column missing in fetched batch.")
            df_processed_batch["LAST_UPDATE_TIME"] = pd.NaT

        num_cols = ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'TRACK']
        for col in num_cols:
            if col in df_processed_batch.columns:
                df_processed_batch[col] = pd.to_numeric(df_processed_batch[col], errors='coerce')

        df_processed_batch.dropna(subset=essential_cols, inplace=True)
        if df_processed_batch.empty:
            print(f"Batch empty after dropping rows with null essential data.")
            return f"{success_msg_base} (No Valid Data Points)"

        str_cols = ['AIRCRAFT_MODEL', 'REGISTRATION', 'ORIGIN', 'DESTINATION', 'FLIGHT_NUMBER', 'FLIGHT_CALLSIGN']
        for col in str_cols:
            if col not in df_processed_batch.columns:
                df_processed_batch[col] = 'N/A'
            else:
                df_processed_batch[col] = df_processed_batch[col].fillna('N/A').astype(str).str.strip()
                for na_val in NA_REPLACE_VALUES:
                    df_processed_batch[col] = df_processed_batch[col].replace(na_val, 'N/A', regex=False)

        if FLIGHT_ID_COLUMN in df_processed_batch.columns:
            df_processed_batch[FLIGHT_ID_COLUMN] = df_processed_batch[FLIGHT_ID_COLUMN].fillna('N/A').astype(str).str.strip()
            for na_val in NA_REPLACE_VALUES:
                df_processed_batch[FLIGHT_ID_COLUMN] = df_processed_batch[FLIGHT_ID_COLUMN].replace(na_val, 'N/A', regex=False)
            df_processed_batch = df_processed_batch[df_processed_batch[FLIGHT_ID_COLUMN] != 'N/A']

        if df_processed_batch.empty:
            print(f"Batch empty after final Flight ID filtering.")
            return f"{success_msg_base} (No Valid Flight IDs)"

        df_old = load_data_from_csv()
        if 'LAST_UPDATE_TIME' in df_old.columns:
            df_old['LAST_UPDATE_TIME'] = pd.to_datetime(df_old['LAST_UPDATE_TIME'], errors='coerce', utc=True)
        else:
            df_old['LAST_UPDATE_TIME'] = pd.NaT

        for col in essential_cols:
            if col not in df_old.columns:
                df_old[col] = pd.NA if col != 'LAST_UPDATE_TIME' else pd.NaT

        df_old.dropna(subset=['LAST_UPDATE_TIME'], inplace=True)

        all_cols = set(df_old.columns) | set(df_processed_batch.columns)
        df_combined = pd.concat([
            df_old.reindex(columns=all_cols),
            df_processed_batch.reindex(columns=all_cols)
            ], ignore_index=True)

        if 'LAST_UPDATE_TIME' in df_combined.columns and not pd.api.types.is_datetime64_any_dtype(df_combined['LAST_UPDATE_TIME']):
            df_combined['LAST_UPDATE_TIME'] = pd.to_datetime(df_combined['LAST_UPDATE_TIME'], errors='coerce', utc=True)

        df_combined.dropna(subset=['LAST_UPDATE_TIME'], inplace=True)

        if not df_combined.empty:
            cutoff_time = datetime.now(PROCESSING_TIMEZONE) - timedelta(hours=PERSISTENCE_HOURS)
            df_persistent = df_combined[df_combined['LAST_UPDATE_TIME'] >= cutoff_time].copy()

            if len(df_persistent) > MAX_STORE_RECORDS:
                print(f"Warning: Exceeded MAX_STORE_RECORDS ({MAX_STORE_RECORDS}). Trimming oldest data.")
                df_persistent = df_persistent.sort_values(by='LAST_UPDATE_TIME', ascending=False).head(MAX_STORE_RECORDS)

        else:
            df_persistent = pd.DataFrame()

        if df_persistent.empty:
            print("No data left after pruning.")
            status_msg = f"No data within {PERSISTENCE_HOURS} hours. Last check: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            # Save empty dataframe to clear the CSV if needed
            df_persistent.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')
            return status_msg

        print(f"Combined data: {len(df_combined)} rows. Persisting {len(df_persistent)} rows (last {PERSISTENCE_HOURS}h).")

        if 'LAST_UPDATE_TIME' in df_persistent.columns:
            df_persistent['LAST_UPDATE_TIME'] = df_persistent['LAST_UPDATE_TIME'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        df_final_persistent = df_persistent.replace({pd.NA: None, pd.NaT: None, np.nan: None})

        # Save to CSV
        try:
            df_final_persistent.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')
            print(f"Data saved to CSV: {CSV_FILE_PATH}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

        status_msg = f"Data updated: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ({len(df_processed_batch)} new pts | {len(df_final_persistent)} total pts stored)"

        return status_msg

    except Exception as e:
        print(f"ERROR in update_data: Unexpected failure - {e}")
        traceback.print_exc()
        return f"{error_msg_base} (Processing Error)"


@callback(
    Output("map-aircraft-filter", "options"),
    Input("interval-component", "n_intervals"), # Changed input
    prevent_initial_call=True
)
def update_aircraft_filter_options(n_intervals): # Removed persistent_data argument
    df = load_data_from_csv()
    if df.empty: return []
    try:
        if "AIRCRAFT_MODEL" not in df.columns: return []
        all_types = df["AIRCRAFT_MODEL"].astype(str).fillna('N/A').replace(NA_REPLACE_VALUES, 'N/A', regex=False).unique()
        valid_types = sorted([t for t in all_types if t != 'N/A'])
        options = [{"label": ac_type, "value": ac_type} for ac_type in valid_types]
        if 'N/A' in all_types: options.append({"label": "N/A", "value": "N/A"})
        return options
    except Exception as e:
        print(f"Filter options Err: {e}")
        return []

@callback(
    Output("flight-table", "data"),
    Output("flight-table", "columns"),
    Output("flight-table", "selected_rows"),
    Input("interval-component", "n_intervals"), # Changed input
    prevent_initial_call=True
)
def update_flight_table(n_intervals): # Removed persistent_data argument
    df_persistent = load_data_from_csv()
    print(f"Updating flight table from {len(df_persistent)} records in CSV.")
    if df_persistent.empty:
        return [], [], []
    try:
        df_persistent = df_persistent.replace({None: pd.NA})

        if 'LAST_UPDATE_TIME' in df_persistent.columns:
            df_persistent['LAST_UPDATE_TIME'] = pd.to_datetime(df_persistent['LAST_UPDATE_TIME'], errors='coerce', utc=True)
        else: df_persistent['LAST_UPDATE_TIME'] = pd.NaT

        if FLIGHT_ID_COLUMN in df_persistent.columns:
            df_persistent[FLIGHT_ID_COLUMN] = df_persistent[FLIGHT_ID_COLUMN].astype(str).fillna('N/A')
        else: df_persistent[FLIGHT_ID_COLUMN] = 'N/A'

        df_persistent.dropna(subset=[FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME'], inplace=True)
        df_persistent = df_persistent[df_persistent[FLIGHT_ID_COLUMN] != 'N/A']

        if df_persistent.empty:
            print("No valid data after initial cleaning for table.")
            return [], [], []

        latest_idx = df_persistent.loc[df_persistent.groupby(FLIGHT_ID_COLUMN)["LAST_UPDATE_TIME"].idxmax()].index
        df_display = df_persistent.loc[latest_idx].copy()
        print(f"Displaying {len(df_display)} latest flight records in table.")

    except Exception as e:
        print(f"Table Update Err: {e}")
        traceback.print_exc()
        return [], [], []

    display_cols_ordered = [
        FLIGHT_ID_COLUMN,'FLIGHT_CALLSIGN','AIRCRAFT_MODEL','REGISTRATION',
        'LATITUDE','LONGITUDE','ALTITUDE','SPEED','TRACK',
        'ORIGIN','DESTINATION','FLIGHT_NUMBER','LAST_UPDATE_TIME'
    ]
    display_cols = [c for c in display_cols_ordered if c in df_display.columns]
    columns = [{"name": c.replace('_', ' ').title(), "id": c} for c in display_cols]

    num_format_options = {
        'LATITUDE': '{:,.4f}', 'LONGITUDE': '{:,.4f}',
        'ALTITUDE': '{:,.0f}', 'SPEED': '{:,.0f}', 'TRACK': '{:,.0f}'
    }
    for col, fmt in num_format_options.items():
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
            df_display[col] = df_display[col].apply(lambda x: fmt.format(x) if pd.notna(x) else None)

    if 'LAST_UPDATE_TIME' in df_display.columns and pd.api.types.is_datetime64_any_dtype(df_display['LAST_UPDATE_TIME']):
        try:
            dt_series = df_display['LAST_UPDATE_TIME']
            if dt_series.dt.tz is None:
                dt_series = dt_series.dt.tz_localize(PROCESSING_TIMEZONE)

            df_display['LAST_UPDATE_TIME'] = dt_series.dt.tz_convert(DISPLAY_TIMEZONE).dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as tz_e:
            print(f"Table TZ formatting Err: {tz_e}")
            df_display['LAST_UPDATE_TIME'] = df_display['LAST_UPDATE_TIME'].astype(str)

    df_dict = df_display[display_cols].replace({pd.NA: None, np.nan: None, pd.NaT: None}).to_dict("records")

    return df_dict, columns, []

@callback(
    Output("flight-map", "figure"),
    Input("interval-component", "n_intervals"), # Changed input
    Input("map-display-mode", "value"),
    Input("map-aircraft-filter", "value"),
    Input("map-altitude-range", "value"),
    Input("flight-table", "derived_virtual_data"),
    prevent_initial_call=True
)
def update_flight_map(n_intervals, map_display_mode, map_selected_aircraft, map_altitude_range, table_derived_data): # Removed persistent_data argument
    df = load_data_from_csv()
    if df.empty:
        return create_empty_map_figure("Waiting for data...")

    print(f"Updating map from {len(df)} records in CSV. Mode: {map_display_mode}")

    try:
        df = df.replace({None: pd.NA})

        essential_cols = ['LATITUDE', 'LONGITUDE', 'LAST_UPDATE_TIME', FLIGHT_ID_COLUMN]
        if not all(c in df.columns for c in essential_cols):
            return create_empty_map_figure("Data Missing Essential Columns")

        num_cols = ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'TRACK']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df['LAST_UPDATE_TIME'] = pd.to_datetime(df['LAST_UPDATE_TIME'], errors='coerce', utc=True)
        df[FLIGHT_ID_COLUMN] = df[FLIGHT_ID_COLUMN].astype(str).fillna('N/A')
        df.dropna(subset=essential_cols, inplace=True)
        df = df[df[FLIGHT_ID_COLUMN] != 'N/A']

        if df.empty: return create_empty_map_figure("No valid points after cleaning")

        df_filtered = df.copy()

        if map_selected_aircraft:
            valid_aircraft_filter = [ac for ac in map_selected_aircraft if ac]
            if valid_aircraft_filter:
                if 'AIRCRAFT_MODEL' in df_filtered.columns:
                    df_filtered['AIRCRAFT_MODEL'] = df_filtered['AIRCRAFT_MODEL'].fillna('N/A').astype(str)
                    df_filtered = df_filtered[df_filtered['AIRCRAFT_MODEL'].isin(valid_aircraft_filter)]
                else:
                    df_filtered = pd.DataFrame()

        if map_altitude_range and 'ALTITUDE' in df_filtered.columns and not df_filtered.empty:
            alt_min, alt_max = map_altitude_range
            alt_series = pd.to_numeric(df_filtered['ALTITUDE'], errors='coerce')
            altitude_condition = alt_series.between(alt_min, alt_max, inclusive='both') | alt_series.isna()
            df_filtered = df_filtered[altitude_condition]

        table_filtered_ids = set()
        if table_derived_data:
            try:
                table_df = pd.DataFrame(table_derived_data);
                if FLIGHT_ID_COLUMN in table_df.columns:
                    table_filtered_ids = set(table_df[FLIGHT_ID_COLUMN].astype(str).unique())
            except Exception as e:
                print(f"Map table filter sync Err: {e}")
        if table_filtered_ids:
            df_filtered = df_filtered[df_filtered[FLIGHT_ID_COLUMN].isin(table_filtered_ids)]

        if df_filtered.empty:
            return create_empty_map_figure("No data matches filters")

        str_cols_map = ['AIRCRAFT_MODEL', 'FLIGHT_CALLSIGN', 'REGISTRATION', 'ORIGIN', 'DESTINATION', 'FLIGHT_NUMBER']
        for c in str_cols_map:
            if c not in df_filtered.columns: df_filtered[c] = 'N/A'
            df_filtered[c] = df_filtered[c].fillna('N/A').astype(str).str.strip().replace(NA_REPLACE_VALUES, 'N/A', regex=False)

        def create_hover(row):
            alt_s = f"{row.get('ALTITUDE'):,.0f}ft" if pd.notna(row.get('ALTITUDE')) else "---"
            spd_s = f"{row.get('SPEED'):,.0f}kt" if pd.notna(row.get('SPEED')) else "---"
            track_s = f"{row.get('TRACK'):.0f}°" if pd.notna(row.get('TRACK')) else "---"
            lat_s = f"{row.get('LATITUDE'):.4f}" if pd.notna(row.get('LATITUDE')) else "N/A"
            lon_s = f"{row.get('LONGITUDE'):.4f}" if pd.notna(row.get('LONGITUDE')) else "N/A"
            callsign = row.get('FLIGHT_CALLSIGN', 'N/A')
            flight_num = row.get('FLIGHT_NUMBER', 'N/A')
            reg = row.get('REGISTRATION', 'N/A')
            model = row.get('AIRCRAFT_MODEL', 'N/A')
            origin = row.get('ORIGIN', 'N/A')
            dest = row.get('DESTINATION', 'N/A')
            return f"<b>{callsign or flight_num or 'N/A'}</b> ({flight_num if callsign != flight_num else ''})<br>Reg: {reg} | Model: {model}<br>Route: {origin} → {dest}<br>Coords: {lat_s}, {lon_s}<br>Alt: {alt_s} | Spd: {spd_s} | Track: {track_s}"

        fig = go.Figure()
        layout = go.Layout(
            mapbox=dict(style=MAPBOX_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
            margin=dict(r=5,t=5,l=5,b=5),
            paper_bgcolor=CHART_PAPER_BG,
            plot_bgcolor=CHART_PLOT_BG,
            legend_title_text='Aircraft Type',
            legend=dict(
                orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.99,
                bgcolor="rgba(44,48,52,0.8)", bordercolor="#6c757d", borderwidth=1,
                font=dict(color="#adb5bd")
            ),
            uirevision=f"{map_display_mode}-{map_selected_aircraft}-{map_altitude_range}-{table_filtered_ids}"
        )

        if map_display_mode == 'current':
            latest_idx = df_filtered.loc[df_filtered.groupby(FLIGHT_ID_COLUMN)["LAST_UPDATE_TIME"].idxmax()].index
            df_plot = df_filtered.loc[latest_idx].copy()

            if df_plot.empty: return create_empty_map_figure("No latest data points found")

            df_plot['hover_text'] = df_plot.apply(create_hover, axis=1)
            df_plot['map_label'] = df_plot.apply(lambda row: f"{row.get('FLIGHT_CALLSIGN', row.get(FLIGHT_ID_COLUMN))}<br>{row.get('ALTITUDE', ''):,.0f}ft" if pd.notna(row.get('ALTITUDE')) else f"{row.get('FLIGHT_CALLSIGN', row.get(FLIGHT_ID_COLUMN))}", axis=1)

            fig_px = px.scatter_mapbox(df_plot,
                lat="LATITUDE", lon="LONGITUDE",
                hover_name=FLIGHT_ID_COLUMN,
                hover_data={"LATITUDE":False, "LONGITUDE":False, "hover_text":True},
                custom_data=['hover_text'],
                size="SPEED",
                color="AIRCRAFT_MODEL",
                text="map_label",
                mapbox_style=MAPBOX_STYLE,
                opacity=MAP_MARKER_OPACITY,
                size_max=MAP_POINT_SIZE_MAX
            )
            for trace in fig_px.data:
                fig.add_trace(trace)

            fig.update_traces(
                textfont=dict(color=ON_MAP_TEXT_COLOR, size=ON_MAP_TEXT_SIZE),
                textposition=ON_MAP_TEXT_POSITION,
                hovertemplate='%{customdata[0]}<extra></extra>'
            )

        elif map_display_mode == 'routes':
            df_plot = df_filtered.sort_values(by=[FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME'])
            if df_plot.empty: return create_empty_map_figure("No route data found")

            df_plot['hover_text'] = df_plot.apply(create_hover, axis=1)

            fig_px = px.line_mapbox(df_plot,
                lat="LATITUDE", lon="LONGITUDE",
                color=FLIGHT_ID_COLUMN,
                mapbox_style=MAPBOX_STYLE,
                hover_name=FLIGHT_ID_COLUMN,
                hover_data={"LATITUDE":False, "LONGITUDE":False, "hover_text":True},
                custom_data=['hover_text']
            )
            for trace in fig_px.data:
                fig.add_trace(trace)

            fig.update_traces(
                line=dict(width=MAP_LINE_WIDTH),
                hovertemplate='%{customdata[0]}<extra></extra>'
            )
            layout.update(showlegend=False)

            # Add circle at the end of each route
            df_last = df_plot.groupby(FLIGHT_ID_COLUMN).last().reset_index()
            df_last['hover_text'] = df_last.apply(create_hover, axis=1)

            fig.add_trace(go.Scattermapbox(
                lat=df_last['LATITUDE'],
                lon=df_last['LONGITUDE'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=df_last[FLIGHT_ID_COLUMN].astype('category').cat.codes, # Use same color as line
                    opacity=0.8
                ),
                text=df_last['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name='Flight Endpoints'
            ))

        else:
            return create_empty_map_figure(f"Unknown display mode: {map_display_mode}")

        fig.update_layout(layout)
        return fig

    except Exception as e:
        print(f"Map generation error: {e}")
        traceback.print_exc()
        return create_empty_map_figure("Error Generating Map")

@callback(
    Output("analytics-total-flights", "children"),
    Output("analytics-avg-speed", "children"),
    Output("analytics-flights-per-day", "figure"),
    Output("analytics-aircraft-types", "figure"),
    Output("analytics-altitude-dist", "figure"),
    Output("analytics-speed-dist", "figure"),
    Output("analytics-top-origins", "figure"),
    Output("analytics-top-routes", "figure"),
    Input("interval-component", "n_intervals"), # Changed input
    Input("flight-table", "derived_virtual_data"),
)
def update_analytics(n_intervals, table_derived_data): # Removed persistent_data argument
    df_persistent = load_data_from_csv()
    no_data_str="-"; no_data_fig=create_empty_analytics_figure();
    default_outputs=(no_data_str, no_data_str, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig)

    if df_persistent.empty:
        return default_outputs

    print(f"Updating analytics from {len(df_persistent)} records in CSV.")

    try:
        df_persistent = df_persistent.replace({None: pd.NA})
        if df_persistent.empty: raise ValueError("Empty persistent data for analytics")

        df_persistent[FLIGHT_ID_COLUMN] = df_persistent[FLIGHT_ID_COLUMN].astype(str).fillna('N/A')
        df_persistent['LAST_UPDATE_TIME'] = pd.to_datetime(df_persistent['LAST_UPDATE_TIME'], errors='coerce', utc=True)
        df_persistent['ALTITUDE'] = pd.to_numeric(df_persistent['ALTITUDE'], errors='coerce')
        df_persistent['SPEED'] = pd.to_numeric(df_persistent['SPEED'], errors='coerce')
        str_cols_an = ['AIRCRAFT_MODEL', 'ORIGIN', 'DESTINATION', 'FLIGHT_CALLSIGN']
        for col in str_cols_an:
            if col in df_persistent.columns:
                df_persistent[col] = df_persistent[col].fillna('N/A').astype(str).replace(NA_REPLACE_VALUES, 'N/A', regex=False)
            else: df_persistent[col] = 'N/A'

        df_persistent.dropna(subset=[FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME'], inplace=True)
        df_persistent = df_persistent[df_persistent[FLIGHT_ID_COLUMN] != 'N/A']
        if df_persistent.empty: raise ValueError("Data empty after essential cleaning")

        table_filtered_ids = set()
        if table_derived_data:
            try:
                if isinstance(table_derived_data, list) and len(table_derived_data) > 0:
                    tbl_df = pd.DataFrame(table_derived_data)
                    if FLIGHT_ID_COLUMN in tbl_df.columns:
                        valid_ids = tbl_df[FLIGHT_ID_COLUMN].dropna().astype(str)
                        table_filtered_ids = set(valid_ids.unique())
            except Exception as e: print(f"Analytics table filter Err: {e}")

        df_processed = df_persistent[df_persistent[FLIGHT_ID_COLUMN].isin(table_filtered_ids)].copy() if table_filtered_ids else df_persistent.copy()
        if df_processed.empty: raise ValueError("Data empty after table filter")

        latest_idx = df_processed.loc[df_processed.groupby(FLIGHT_ID_COLUMN)["LAST_UPDATE_TIME"].idxmax()].index
        df_latest = df_processed.loc[latest_idx].copy()
        if df_latest.empty: raise ValueError("No latest records found for analytics")

        print(f"Calculating analytics based on {len(df_latest)} latest flight records.")

        total_flights_str = f"{df_latest[FLIGHT_ID_COLUMN].nunique()}"
        avg_speed = df_latest['SPEED'].mean()
        avg_speed_str = f"{avg_speed:,.0f}" if pd.notna(avg_speed) else "-"

        chart_margin = dict(l=20, r=20, t=30, b=20)
        chart_layout_args = dict(margin=chart_margin, paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG, font_color=CHART_FONT_COLOR, hoverlabel=dict(bgcolor="#444", font_color="white"))
        color_sequence = px.colors.qualitative.Prism_r

        df_processed['DATE'] = df_processed['LAST_UPDATE_TIME'].dt.tz_convert(DISPLAY_TIMEZONE).dt.date
        flights_per_day = df_processed.groupby('DATE')[FLIGHT_ID_COLUMN].nunique().reset_index(name='Count').sort_values('DATE')
        flights_per_day['DATE_STR'] = flights_per_day['DATE'].astype(str)
        fig_day = create_empty_analytics_figure("No Daily Data") if flights_per_day.empty else px.bar(flights_per_day, x='DATE_STR', y='Count', template=PLOTLY_TEMPLATE, color='DATE_STR', color_discrete_sequence=color_sequence)
        fig_day.update_layout(yaxis_title="# Flights", xaxis_title=f"Date ({DISPLAY_TIMEZONE.zone})", showlegend=False, **chart_layout_args)

        top_types = df_latest['AIRCRAFT_MODEL'].value_counts().nlargest(10).reset_index(name='Count').rename(columns={'index': 'AIRCRAFT_MODEL'})
        fig_types = create_empty_analytics_figure("No Type Data") if top_types.empty else px.bar(top_types, x='Count', y='AIRCRAFT_MODEL', orientation='h', template=PLOTLY_TEMPLATE, color='AIRCRAFT_MODEL', color_discrete_sequence=color_sequence)
        fig_types.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="# Flights", yaxis_title=None, showlegend=False, **chart_layout_args)

        alt_data = df_latest['ALTITUDE'].dropna()
        fig_alt = create_empty_analytics_figure("No Alt Data") if alt_data.empty else px.histogram(alt_data, x="ALTITUDE", nbins=12, template=PLOTLY_TEMPLATE, color_discrete_sequence=[color_sequence[1]])
        fig_alt.update_layout(yaxis_title="# Flights", xaxis_title="Altitude (ft)", bargap=0.1, **chart_layout_args)

        speed_data = df_latest['SPEED'].dropna()
        fig_speed = create_empty_analytics_figure("No Speed Data") if speed_data.empty else px.histogram(speed_data, x="SPEED", nbins=12, template=PLOTLY_TEMPLATE, color_discrete_sequence=[color_sequence[2]])
        fig_speed.update_layout(yaxis_title="# Flights", xaxis_title="Speed (kts)", bargap=0.1, **chart_layout_args)

        origin_counts = df_latest[df_latest['ORIGIN'] != 'N/A']['ORIGIN'].value_counts().nlargest(5).reset_index(name='Count').rename(columns={'index': 'ORIGIN'})
        fig_origins = create_empty_analytics_figure("No Origin Data") if origin_counts.empty else px.bar(origin_counts, x='Count', y='ORIGIN', orientation='h', template=PLOTLY_TEMPLATE, color='ORIGIN', color_discrete_sequence=color_sequence)
        fig_origins.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="# Flights", yaxis_title=None, showlegend=False, **chart_layout_args)

        df_latest['ROUTE'] = df_latest['ORIGIN'] + ' → ' + df_latest['DESTINATION']
        valid_routes = df_latest[(df_latest['ORIGIN'] != 'N/A') & (df_latest['DESTINATION'] != 'N/A')]
        route_counts = valid_routes['ROUTE'].value_counts().nlargest(5).reset_index(name='Count').rename(columns={'index': 'ROUTE'})
        fig_routes = create_empty_analytics_figure("No Route Data") if route_counts.empty else px.bar(route_counts, x='Count', y='ROUTE', orientation='h', template=PLOTLY_TEMPLATE, color='ROUTE', color_discrete_sequence=color_sequence)
        fig_routes.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="# Flights", yaxis_title=None, showlegend=False, **chart_layout_args)

        for fig in [fig_day, fig_types, fig_alt, fig_speed, fig_origins, fig_routes]:
            if isinstance(fig, go.Figure):
                fig.update_layout(modebar={'remove': ['toimage', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']})

        return total_flights_str, avg_speed_str, fig_day, fig_types, fig_alt, fig_speed, fig_origins, fig_routes

    except ValueError as ve:
        print(f"Analytics Calc Info: {ve}")
        return default_outputs
    except Exception as e:
        print(f"Analytics Calc Error: {e}")
        traceback.print_exc()
        return default_outputs

@callback(
    Output("post-feedback", "children"),
    Input("post-button", "n_clicks"),
    State("flight-table", "selected_rows"),
    State("flight-table", "data"),
    prevent_initial_call=True
)
def post_selected_flights(n_clicks, selected_indices, table_data):
    if n_clicks == 0 or not selected_indices:
        return ""

    if not table_data:
        return dbc.Alert("No data in table to select from.", color="warning", dismissable=True)

    try:
        selected_data = [table_data[i] for i in selected_indices]
    except IndexError:
        return dbc.Alert("Selection index out of bounds. Please refresh.", color="danger", dismissable=True)
    except TypeError:
        return dbc.Alert("Table data is not in the expected format.", color="danger", dismissable=True)

    if not selected_data:
        return dbc.Alert("No rows selected for posting.", color="warning", dismissable=True)

    print(f"Posting {len(selected_data)} selected rows (from table view) to {POST_API_URL}")

    try:
        response = requests.post(
            POST_API_URL,
            json=selected_data,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        response.raise_for_status()

        try:
            resp_json = response.json()
            msg = resp_json.get('message', f"Successfully posted {len(selected_data)} records.")
            details = ""
            if 'processed' in resp_json or 'errors' in resp_json:
                processed_count = resp_json.get('processed', '?')
                error_count = resp_json.get('errors', '?')
                details=f" (Server Response: Processed: {processed_count}, Errors: {error_count})"
            success_msg = f"{msg}{details}"
        except requests.exceptions.JSONDecodeError:
            success_msg = f"Posted {len(selected_data)} records. Status: {response.status_code} (Non-JSON response)"

        print(f"Post successful: {success_msg}")
        return dbc.Alert(success_msg, color="success", dismissable=True, duration=5000)

    except requests.exceptions.Timeout:
        print("Post Error: Request timed out.")
        return dbc.Alert("Request timed out trying to post data.", color="danger", dismissable=True)
    except requests.exceptions.ConnectionError:
        print("Post Error: Could not connect to the server.")
        return dbc.Alert("Connection error: Could not reach the posting API.", color="danger", dismissable=True)
    except requests.exceptions.HTTPError as http_err:
        err_msg = f"Post Error: HTTP {http_err.response.status_code} Error."
        err_detail = ""
        try:
            err_detail = f" Response: {http_err.response.text[:200]}..."
        except Exception: pass
        print(f"{err_msg}{err_detail}")
        return dbc.Alert(f"{err_msg}{err_detail}", color="danger", dismissable=True, duration=9000)
    except requests.exceptions.RequestException as req_err:
        err_msg = f"Post Error: {req_err}"
        print(err_msg)
        return dbc.Alert(f"An error occurred during the request: {err_msg}", color="danger", dismissable=True)
    except Exception as e:
        print(f"Post Error: Unexpected failure - {e}")
        traceback.print_exc()
        return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", dismissable=True)

@callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("flight-table", "data"),
    prevent_initial_call=True
)
def download_table_csv(n_clicks, table_data):
    if n_clicks == 0 or not table_data:
        return None

    if isinstance(table_data, list) and len(table_data) > 0:
        try:
            df_download = pd.DataFrame(table_data)
            ts = datetime.now(DISPLAY_TIMEZONE).strftime("%Y%m%d_%H%M%S")
            filename = f"flight_data_snapshot_{ts}.csv"
            return dcc.send_data_frame(df_download.to_csv, filename=filename, index=False, encoding='utf-8')

        except Exception as e:
            print(f"CSV Export Error: {e}")
            return None
    else:
        print("No table data available for CSV export.")
        return None

@callback(
    Output("insights-modal", "is_open"), Output("insights-modal-body", "children"),
    Input("insights-button", "n_clicks"), Input("insights-modal-close", "n_clicks"),
    State("insights-modal", "is_open"),
    Input("interval-component", "n_intervals"), # Changed input
    State("flight-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def toggle_and_load_insights(
    btn_open_clicks, btn_close_clicks, is_open,
    n_intervals, table_derived_data
    ):
    trigger_id = ctx.triggered_id
    if trigger_id == "insights-modal-close" and is_open:
        return False, dash.no_update

    if trigger_id == "insights-button":
        df = load_data_from_csv()
        if df.empty:
            return True, html.Div("No flight data available.", className="text-warning")
        if helpers is None:
            return True, html.Div("Error: Insights helper module not loaded.", className="text-danger")

        try:
            print("[Insights] Preparing data based on table filter...")
            df = df.replace({None: np.nan})

            required = [FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME', 'ALTITUDE', 'SPEED', 'AIRCRAFT_MODEL', 'ORIGIN', 'DESTINATION']
            if not all(c in df.columns for c in required):
                missing = [c for c in required if c not in df.columns]
                print(f"Insights Prep Err: Missing required columns in CSV data: {missing}")
                return True, html.Div(f"Data Error for Insights: Missing {', '.join(missing)}", className="text-danger")

            df[FLIGHT_ID_COLUMN] = df[FLIGHT_ID_COLUMN].astype(str)
            df['LAST_UPDATE_TIME'] = pd.to_datetime(df['LAST_UPDATE_TIME'], errors='coerce')
            df.dropna(subset=[FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME'], inplace=True)
            if df.empty: raise ValueError("No valid time/id data in CSV for insights")

            table_filtered_ids = set()
            df_processed = df

            if table_derived_data:
                try:
                    if isinstance(table_derived_data, list) and len(table_derived_data) > 0:
                        tbl_df = pd.DataFrame(table_derived_data)
                        if FLIGHT_ID_COLUMN in tbl_df.columns:
                            valid_ids = tbl_df[FLIGHT_ID_COLUMN].dropna().astype(str)
                            table_filtered_ids = set(valid_ids.unique())
                            print(f"[Insights] Applying table filter, {len(table_filtered_ids)} unique IDs found.")
                except Exception as e:
                    print(f"Insights: Error processing table derived data: {e}")

            if table_filtered_ids:
                df_processed = df[df[FLIGHT_ID_COLUMN].isin(table_filtered_ids)].copy()

            if df_processed.empty:
                print("[Insights] No data remains after applying table filter (or filter was empty).")
                return True, html.Div("No flights match the current table filter to get insights.", className="text-info")

            print("[Insights] Getting latest record for each filtered flight...")
            if not pd.api.types.is_datetime64_any_dtype(df_processed['LAST_UPDATE_TIME']):
                df_processed['LAST_UPDATE_TIME'] = pd.to_datetime(df_processed['LAST_UPDATE_TIME'], errors='coerce')
            df_processed.dropna(subset=['LAST_UPDATE_TIME'], inplace=True)
            if df_processed.empty: raise ValueError("No valid time data after filter dropna")

            try:
                latest_indices = df_processed.loc[df_processed.groupby(FLIGHT_ID_COLUMN)['LAST_UPDATE_TIME'].idxmax()].index
                df_model_input = df_processed.loc[latest_indices].copy()
            except KeyError:
                print("[Insights] Groupby/idxmax failed, likely no valid groups.")
                return True, html.Div("Could not process filtered data for insights.", className="text-warning")

            if df_model_input.empty:
                print("[Insights] No latest records found for filtered data.")
                return True, html.Div("No specific flight records found for insights after filtering.", className="text-info")

            print(f"[Insights] Sending {len(df_model_input)} unique flight records (latest state) to model.")
            csv_buffer = io.StringIO()
            df_model_input.to_csv(csv_buffer, index=False, na_rep='')
            csv_as_string = csv_buffer.getvalue()
            csv_buffer.close()

        except Exception as e:
            print(f"Insights Data Prep Error (Filtered): {e}"); traceback.print_exc()
            return True, html.Div(f"Error preparing filtered data for insights: {e}", className="text-danger")

        try:
            print("[Insights] Calling helper function...")
            raw_model_output = helpers.insights_for_flight_data(csv_as_string)
            model_output_dict = None

            if isinstance(raw_model_output, dict): model_output_dict = raw_model_output
            elif isinstance(raw_model_output, str):
                processed_string = raw_model_output.strip()
                if processed_string.startswith(codecs.BOM_UTF8.decode('utf-8')): processed_string = processed_string[len(codecs.BOM_UTF8.decode('utf-8')):]
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', processed_string)
                if json_match: processed_string = json_match.group(1).strip()
                try: model_output_dict = json.loads(processed_string)
                except json.JSONDecodeError:
                    try: model_output_dict = ast.literal_eval(processed_string); assert isinstance(model_output_dict, dict)
                    except Exception as e_p: print(f"Insights Parse Err: {e_p}"); print(repr(raw_model_output)); return True, dbc.Alert(f"Parse Error: {e_p}", color="danger")
            else: return True, dbc.Alert(f"Unexpected model type: {type(raw_model_output)}", color="danger")
            if not model_output_dict: return True, dbc.Alert("Invalid structure from model", color="warning")

            insights = model_output_dict.get("insights", [])
            filtered_flights = model_output_dict.get("filtered_flights", [])
            insight_items = [dbc.ListGroupItem([html.H6(i.get("title", "?"), className="mb-1"), html.P(i.get("description", ""), className="mb-1 small text-muted")]) for i in insights if isinstance(i, dict)]
            insights_comp = dbc.ListGroup(insight_items, flush=True) if insight_items else html.P("No insights generated.", className="text-muted")

            flights_comp = None
            valid_flights = [f for f in filtered_flights if isinstance(f, dict)] if filtered_flights else []
            if valid_flights:
                try:
                    ff_cols = [{"name": k.replace('_', ' ').title(), "id": k} for k in valid_flights[0].keys()]
                    flights_comp = dash_table.DataTable(
                        data=valid_flights, columns=ff_cols,
                        style_header={'backgroundColor': '#2c3034', 'color': 'white', 'fontWeight': 'bold', 'border': '1px solid #454d55'},
                        style_cell={'textAlign': 'left', 'padding': '5px', 'backgroundColor': '#212529', 'color': 'white', 'border': '1px solid #454d55'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#2c3034'}],
                        page_size=5, style_table={'overflowX': 'auto', 'margin-top': '15px'}
                    )
                except Exception as table_ex:
                    print(f"Insights Table Err: {table_ex}")
                    flights_comp = html.P("Error displaying highlighted flights.", className="text-danger mt-3")
            else:
                flights_comp = html.P("No specific flights highlighted by the model.", className="text-muted mt-3")

            modal_body_content = html.Div([
                html.H5("Model Insights:", className="text-info"), insights_comp,
                html.Hr(),
                html.H5("Flights Highlighted by Model:", className="text-info"), flights_comp
            ])
            print("[Insights] Successfully processed model output.")
            return True, modal_body_content

        except Exception as e:
            print(f"Insights Model Call/Processing Err: {e}"); traceback.print_exc()
            return True, html.Div(f"Error during insight processing: {e}", className="text-danger")

    return is_open, dash.no_update

# --- Run the App ---
if __name__ == "__main__":
    print(f"Starting Dash app on http://127.0.0.1:5178")
    print("-" * 50)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Initializing Dash Application...")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Data Source: CSV file: {CSV_FILE_PATH}")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Update Interval (API to CSV): {FETCH_INTERVAL_SECONDS} seconds")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Data Persistence: Storing last {PERSISTENCE_HOURS} hours in CSV")
    print("-" * 50)
    print("Starting Dash server...")
    print("(Use Ctrl+C to stop)")
    print("-" * 50)
    try:
        app.run(debug=True, host= "0.0.0.0", port=5998, use_reloader=False)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down Dash server...")
    finally:
        pass
