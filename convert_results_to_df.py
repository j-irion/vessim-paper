import argparse
import os
import pandas as pd
import yaml
import numpy as np

total_embodied_carbon_intensity = {
    "wind": 349,  # kgCO2/kWp over lifetime
    "solar": 412,  # kgCO2/kWp over lifetime
    "battery": 74,  # kgCO2/kWh capacity
}

embodied_carbon_intensity = {
    "wind": 12,  # gCO2/kWh produced
    "solar": 19,  # gCO2/kWh produced
    "battery": 74,  # kgCO2/kWh capacity
}

dt_h = 1.0 / 60.0


def compute_instantaneous_coverage(df: pd.DataFrame) -> pd.Series:
    """
    For each row, compute coverage = (renewable energy + battery discharge energy)
    / total load energy, expressed as percent and clipped to [0,100].
    Aligns the battery drop with the same interval’s load.
    """
    P_load = np.abs(df["total_consumption"])
    P_renew = df["total_renewable_power"]

    E_load = P_load * dt_h
    E_renew = P_renew * dt_h

    if "storage.charge_level" in df.columns:
        dSOC_wh = df["storage.charge_level"].diff().fillna(0)
        E_batt = (-dSOC_wh).clip(lower=0).shift(-1).fillna(0)
    else:
        E_batt = pd.Series(0.0, index=df.index)

    cov = (E_renew + E_batt) / E_load.replace({0: np.nan})
    cov = cov.clip(lower=0, upper=1) * 100
    return cov


def convert_results_to_df(results_folder: str, location: str):
    records = []

    for folder_idx, first_lvl in enumerate(os.scandir(results_folder)):
        if not first_lvl.is_dir():
            continue

        print(f"Processing folder {folder_idx}: {first_lvl.name}")
        for sub in os.scandir(first_lvl.path):
            if not sub.is_dir() or sub.name.startswith("."):
                continue

            print(f"  Analyzing {sub.name}")
            folder = os.path.expanduser(sub.path)
            cfg_path = os.path.join(folder, ".hydra", "config.yaml")
            data_path = os.path.join(folder, "merged_data.csv")

            with open(cfg_path, "r") as f:
                config = yaml.safe_load(f)

            if "capacity_tuple" in config:
                wind_cap, solar_cap, batt_cap = map(int, config["capacity_tuple"].split(","))
            else:
                wind_cap = config.get("wind_system_capacity", 0)
                solar_cap = config.get("solar_system_capacity", 0)
                batt_cap = config.get("battery_capacity", 0)

            initial_embodied_gCO2 = (
                wind_cap * total_embodied_carbon_intensity["wind"]
                + solar_cap * total_embodied_carbon_intensity["solar"]
                + batt_cap * total_embodied_carbon_intensity["battery"]
            ) * 1000  # kg → g

            df = pd.read_csv(data_path)

            total_wind_kwh = (df["actor_states.Wind.p"].sum() / 1000) * dt_h
            total_solar_kwh = (df["actor_states.Solar.p"].sum() / 1000) * dt_h
            emb_wind = embodied_carbon_intensity["wind"] * total_wind_kwh
            emb_solar = embodied_carbon_intensity["solar"] * total_solar_kwh
            emb_batt = batt_cap * (embodied_carbon_intensity["battery"] * 1000)
            embodied_carbon_g = emb_wind + emb_solar + emb_batt

            cov_series = compute_instantaneous_coverage(df)
            coverage_pct = cov_series.mean() if not cov_series.empty else 0.0

            P_load = np.abs(df["total_consumption"])
            P_renew = df["total_renewable_power"]
            E_load = P_load * dt_h
            E_renew = P_renew * dt_h
            cov_nobatt = (E_renew / E_load.replace({0: np.nan})).clip(0, 1) * 100
            coverage_nobatt_pct = cov_nobatt.mean() if len(cov_nobatt) > 0 else 0.0

            dSOC_wh = (
                df["storage_state.charge_level"].diff().fillna(0)
                if "storage_state.charge_level" in df.columns
                else pd.Series(0, index=df.index)
            )
            E_batt = (-dSOC_wh).clip(lower=0)
            E_charge = dSOC_wh.clip(lower=0)
            E_excess = (E_renew - E_load - E_charge).clip(lower=0)
            E_grid = np.maximum(E_load - (E_renew + E_batt), 0)

            total_batt_discharge = E_batt.sum() / 1000.0
            total_excess = E_excess.sum() / 1000.0
            total_grid = E_grid.sum() / 1000.0

            CI = df["carbon_intensity"].values
            E_nonren = np.maximum(E_load - (E_renew + E_batt), 0)
            op_emissions = ((E_nonren / 1000.0) * CI).sum()

            records.append(
                {
                    "location": location,
                    "wind": wind_cap,
                    "solar": solar_cap,
                    "battery": batt_cap,
                    "operational_emissions_total_g": op_emissions,
                    "embodied_emissions_initial_g": initial_embodied_gCO2,
                    "embodied_emissions_g": embodied_carbon_g,
                    "coverage_pct": coverage_pct,
                    "coverage_nobatt_pct": coverage_nobatt_pct,
                    "total_battery_discharge_kwh": total_batt_discharge,
                    "total_excess_renewable_kwh": total_excess,
                    "total_grid_draw_kwh": total_grid,
                }
            )

    results_df = pd.DataFrame.from_records(records)
    results_df.to_csv(f"{location}_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert simulation results into summary DataFrame"
    )
    parser.add_argument("-d", "--results_folder", required=True, help="Path to the results folder")
    parser.add_argument("-l", "--location", required=True, help="Location identifier")
    args = parser.parse_args()

    convert_results_to_df(args.results_folder, args.location)
