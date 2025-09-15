import json
import math
import pandas as pd
import PySAM.Windpower
import PySAM.Pvwattsv8
from vessim.signal import Trace
from pathlib import Path
from typing import Optional

def sam_to_trace(
    model: str,
    weather_file: str,
    *,
    config_file: str | None = None,
    config_object: dict | None = None,
    column_name: str = "power_W",
) -> Trace:
    # --- guards (match your mutual-exclusive semantics) ---
    if (config_file is None) == (config_object is None):
        raise ValueError("Either 'config_file' or 'config_object' must be provided (but not both).")

    # --- build SAM model exactly like before ---
    skiprows = 1
    if model == "Windpower":
        sam = PySAM.Windpower.default("WindPowerNone")
        sam.Resource.wind_resource_filename = weather_file
    elif model == "Pvwattsv8":
        sam = PySAM.Pvwattsv8.default("PVWattsNone")
        sam.SolarResource.solar_resource_file = weather_file
        skiprows = 2
    else:
        raise ValueError(f"Model '{model}' not supported.")

    # --- timestamp index from weather CSV (unchanged) ---
    df_weather = pd.read_csv(weather_file, skiprows=skiprows)
    df_weather["Datetime"] = pd.to_datetime(
        df_weather[["Year", "Month", "Day", "Hour", "Minute"]]
    )
    df_weather.set_index("Datetime", inplace=True)

    # --- load & apply config ---
    sam_cfg = json.load(open(config_file, "r", errors="replace")) if config_file else config_object
    for k, v in sam_cfg.items():
        if k not in ("number_inputs", "wind_resource_filename", "solar_resource_file"):
            try:
                sam.value(k, v)
            except Exception as e:
                print(f"Could not set SAM parameter '{k}': {e}")

    # --- run SAM once ---
    sam.execute()

    # --- system_capacity==0 => zero series (match your early return) ---
    try:
        cap = sam.value("system_capacity")
    except Exception:
        cap = None
    if cap == 0:
        actual = pd.DataFrame({column_name: 0.0}, index=df_weather.index)
        return Trace(actual=actual, fill_method="ffill", column=column_name)  # ffill matches fallback. :contentReference[oaicite:3]{index=3}

    # --- get power from Outputs.gen and scale exactly like your code ---
    if not hasattr(sam.Outputs, "gen"):
        raise RuntimeError("SAM Outputs.gen not found; adjust extraction to your model outputs.")
    vals = list(sam.Outputs.gen)
    # align length to timestamps (truncate/exact; you can pad if you prefer)
    n = min(len(vals), len(df_weather.index))
    series = pd.Series(vals[:n], index=df_weather.index[:n], name=column_name) * 1000.0

    actual = series.to_frame()

    # --- wrap in Vessim Trace with ffill (your “last known value” behavior) ---
    return Trace(actual=actual, fill_method="ffill", column=column_name)  # public, library-only. :contentReference[oaicite:4]{index=4}


def file_to_trace(
    file_path: str | Path,
    unit: Optional[str] = "W",
    date_format: Optional[str] = None,
    name: Optional[str] = None,
    invert: bool = False,
    column_name: Optional[str] = None,
) -> Trace:
    """
    Create a Vessim Trace from a two-column CSV like your FileSignal.

    CSV expectations (same as your class):
      - two columns: time, power
      - header present but skipped (skiprows=1), we assign column names ourselves
      - 'time' is parsed with optional strptime 'date_format'
    """
    file_path = Path(file_path)

    # --- 1) load CSV exactly like FileSignal ---
    df = pd.read_csv(file_path, names=["time", "power"], skiprows=1)

    if date_format:
        df["time"] = pd.to_datetime(df["time"], format=date_format)
    else:
        df["time"] = pd.to_datetime(df["time"])

    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)  # ascending

    # uniqueness + monotonic (after sort); duplicates will still fail
    if not df.index.is_unique:
        raise ValueError("The time index must be unique.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("The time index must be monotonic (increasing).")

    # --- 2) convert to Watts exactly like your switch ---
    def _to_watts(power: float, unit: str) -> float:
        if unit == "W":
            return float(power)
        elif unit == "kW":
            return float(power) * 1e3
        elif unit == "MW":
            return float(power) * 1e6
        else:
            raise ValueError(f"Unknown unit: {unit}")

    df["power_W"] = df["power"].astype(float).map(lambda x: _to_watts(x, unit or "W"))
    if invert:
        df["power_W"] = -df["power_W"]

    # choose column name (defaults to "power_W" unless you provide one)
    col = column_name or "power_W"

    actual = df[[ "power_W" ]].rename(columns={"power_W": col})

    # --- 3) wrap as a Trace with forward-fill between stamps (your asof behavior) ---
    # If you later add a forecast DataFrame, pass it as Trace(actual, forecast, ...)
    return Trace(actual=actual, fill_method="ffill", column=col, repr_=name)


def automatic_farm_layout(
    desired_farm_size: float, wind_turbine_kw_rating: float, wind_turbine_rotor_diameter: float
):
    num = math.floor(desired_farm_size / wind_turbine_kw_rating)
    if num <= 1:
        num = 1
    num_turbines = num

    x = [0] * num_turbines
    y = [0] * num_turbines

    rows = math.floor(math.sqrt(num_turbines))
    cols = num_turbines / rows
    while rows * math.floor(cols) != num_turbines:
        rows -= 1
        cols = num_turbines / rows

    spacing_x = 8 * wind_turbine_rotor_diameter
    spacing_y = 8 * wind_turbine_rotor_diameter

    x[0] = 0
    y[0] = 0

    for i in range(1, num_turbines):
        x[i] = (i - cols * math.floor(i / cols)) * spacing_x
        y[i] = math.floor(i / cols) * spacing_y

    return {
        "wind_farm_xCoordinates": x,
        "wind_farm_yCoordinates": y,
    }