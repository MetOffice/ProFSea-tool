import argparse
import json
from pathlib import Path

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.progress import track
import xarray as xr

# --- NEW IMPORTS ---
from profsea.components.core.global_model import Global
from profsea.components.global_ import (
    AntarcticaISMIP6,
    LandwaterAR6,
    GreenlandAR6,
    ThermalExpansion,
    Glacier,
)
from profsea.utils import sample_members_2D

console = Console()


def index_df(df: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    meta_cols = [
        "ensemble_member",
        "scenario",
        "region",
        "variable",
        "unit",
        "climate_model",
    ]
    existing_meta = [c for c in meta_cols if c in df.columns]
    df_indexed = df.set_index(existing_meta)

    years = df_indexed.columns.astype(str).str[:4].astype(int)
    df_indexed.columns = years

    mask_full = (years >= 1750) & (years <= 2300)
    df_indexed = df_indexed.loc[:, mask_full]

    years_sliced = df_indexed.columns
    baseline_mask = (years_sliced >= baseline_start) & (years_sliced <= baseline_end)

    if not baseline_mask.any():
        raise ValueError(
            f"No years found in baseline range {baseline_start}-{baseline_end}"
        )

    baseline_means = df_indexed.loc[:, baseline_mask].mean(axis=1)
    df_anom = df_indexed.sub(baseline_means, axis=0)
    years_final = df_anom.columns
    mask_final = (years_final >= 2006) & (years_final <= 2300)

    df_final = df_anom.loc[:, mask_final].reset_index()
    return df_final


def df_to_arr(df, scenario_order):
    meta_cols = [
        "ensemble_member",
        "scenario",
        "region",
        "variable",
        "unit",
        "climate_model",
    ]
    existing_meta = [c for c in meta_cols if c in df.columns]

    df = df.set_index(existing_meta)
    array = []
    for scenario in scenario_order:
        try:
            mask = df.index.get_level_values("scenario") == scenario
            group_df = df.loc[mask]
        except KeyError:
            raise ValueError(f"Scenario '{scenario}' not found in the input DataFrame.")

        if group_df.empty:
            raise ValueError(f"No data found for scenario '{scenario}'")

        group_sorted = group_df.sort_index(level="ensemble_member")
        scenario_data = group_sorted.values
        array.append(scenario_data)

    array = np.stack(array, axis=0)
    array = array.transpose(2, 0, 1)  # (n_scenarios, n_members, n_time)
    return array


def load_magicc_forcing(
    input_path: str, scenarios: list, baseline_start: int, baseline_end: int
) -> tuple[np.ndarray]:
    df = pd.read_csv(input_path, index_col=0)

    tas_condition = (df["variable"] == "Surface Air Temperature Change") & (
        df["scenario"].isin(scenarios)
    )
    ohc_condition = (df["variable"] == "Heat Content|Ocean") & (
        df["scenario"].isin(scenarios)
    )
    tas_df = df.loc[tas_condition]
    ohc_df = df.loc[ohc_condition]

    tas_df = index_df(tas_df, baseline_start, baseline_end)
    ohc_df = index_df(ohc_df, baseline_start, baseline_end)
    tas_arr = df_to_arr(tas_df, scenarios)
    ohc_arr = df_to_arr(ohc_df, scenarios) * 1e21

    # Trim between 2006 and 2300 just to be safe
    # The years are after the 6th column, so we can slice by column names
    tas_arr = tas_arr[:, :, (tas_df.columns[6] >= 2006) & (tas_df.columns[6] <= 2300)]
    ohc_arr = ohc_arr[:, :, (ohc_df.columns[6] >= 2006) & (ohc_df.columns[6] <= 2300)]

    # These are now of shape (time, scenario, 1, member) - we want (scenario, member, time)
    tas_arr = tas_arr.squeeze().transpose(1, 2, 0)
    ohc_arr = ohc_arr.squeeze().transpose(1, 2, 0)
    return tas_arr, ohc_arr


def load_fair_forcing(
    input_path: str, scenarios: list, baseline_start: int, baseline_end: int
) -> tuple[np.ndarray]:
    tas_path = Path(input_path) / "tas.nc"
    ohc_path = Path(input_path) / "ohc.nc"

    tas = xr.load_dataarray(tas_path)
    ohc = xr.load_dataarray(ohc_path)

    tas_baseline = tas.loc[
        dict(timebounds=np.arange(baseline_start, baseline_end + 1))
    ].mean("timebounds")
    ohc_baseline = ohc.loc[
        dict(timebounds=np.arange(baseline_start, baseline_end + 1))
    ].mean(["timebounds"])

    tas = tas.loc[dict(timebounds=np.arange(2006, 2301))] - tas_baseline
    ohc = ohc.loc[dict(timebounds=np.arange(2006, 2301))] - ohc_baseline

    tas = tas.sel(scenario=scenarios).transpose("scenario", "config", "timebounds")
    ohc = ohc.sel(scenario=scenarios).transpose("scenario", "config", "timebounds")
    return tas.values, ohc.values


def run_fair(
    baseline_start: int, baseline_end: int, scenarios: list, args: argparse.Namespace
) -> tuple[np.ndarray]:
    f = FAIR()
    f.define_time(1750, 2300, 1)
    f.define_scenarios(scenarios)
    species, properties = read_properties(
        "../data/fair/fair-parameters/species_configs_properties_1.4.1.csv"
    )
    f.define_species(species, properties)
    f.ch4_method = "Thornhill2021"
    df_configs = pd.read_csv(
        "../data/fair/fair-parameters/calibrated_constrained_parameters_1.4.1.csv",
        index_col=0,
    )
    f.define_configs(df_configs.index)
    f.allocate()

    if args.emissions_file:
        f.fill_from_csv(
            emissions_file=args.emissions_file, forcing_file=args.forcing_file
        )
    else:
        f.fill_from_rcmip()

    f.fill_species_configs(
        "../data/fair/fair-parameters/species_configs_properties_1.4.1.csv"
    )
    f.override_defaults(
        "../data/fair/fair-parameters/calibrated_constrained_parameters_1.4.1.csv"
    )
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)

    f.run()

    tas_baseline = f.temperature.loc[
        dict(layer=0, timebounds=np.arange(baseline_start, baseline_end + 1))
    ].mean()
    ohc_baseline = f.ocean_heat_content_change.loc[
        dict(timebounds=np.arange(baseline_start, baseline_end + 1))
    ].mean()

    tas = (
        f.temperature.loc[dict(layer=0, timebounds=np.arange(2006, 2301))]
        - tas_baseline
    )
    ohc = (
        f.ocean_heat_content_change.loc[dict(timebounds=np.arange(2006, 2301))]
        - ohc_baseline
    )

    tas = tas.sel(scenario=scenarios).transpose("scenario", "config", "timebounds")
    ohc = ohc.sel(scenario=scenarios).transpose("scenario", "config", "timebounds")
    return tas.values, ohc.values


def plot_samples(tas: np.ndarray, ohc: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 6), layout="constrained")

    ax = fig.add_subplot(121)
    ax.plot(tas.T, color="seagreen", alpha=0.05)
    ax.set_xlabel("Simulation years")
    ax.set_ylabel("GMST ($\degree$C)")
    ax.plot(np.arange(tas.shape[1]), np.median(tas, axis=0), color="black")

    ax = fig.add_subplot(122)
    ax.plot(ohc.T, color="seagreen", alpha=0.05)
    ax.set_xlabel("Simulation years")
    ax.set_ylabel("OHC (J)")

    fig.savefig("forcing.png", dpi=200)
    plt.show()
    plt.close()


def process_global_ensemble(components: list, percentiles: list, scenario: str) -> None:
    for comp, data in components.items():
        sampled_ensemble = sample_members_2D(data, percentiles)
        components[comp] = sampled_ensemble
    return components


def save_to_netcdf(components: dict, filename: str) -> None:
    scenarios = list(components.keys())
    comp_names = list(components[scenarios[0]].keys())
    sample = components[scenarios[0]][comp_names[0]]
    n_member, n_time = sample.shape

    years = np.arange(2006, 2006 + n_time)
    members = np.arange(n_member)
    data_vars = {}
    for comp in comp_names:
        stacked_data = np.stack([components[s][comp] for s in scenarios], axis=0)
        data_vars[comp] = xr.DataArray(
            data=stacked_data,
            dims=["scenario", "member", "year"],
            coords={"scenario": scenarios, "member": members, "year": years},
            attrs={"units": "m", "description": f"Sea level contribution from {comp}"},
        )

    ds = xr.Dataset(data_vars)
    ds.attrs = {
        "title": "ProFSea GMSLR Projections",
        "source": "FAIR v2.2 + ProFSea Emulator",
    }

    encoding = {var: {"zlib": True, "complevel": 5} for var in data_vars}
    ds.to_netcdf(filename, encoding=encoding)
    console.log(f"Successfully saved full ensemble to {filename}")


def plot_component(
    ax: plt.Axes,
    component_dict: dict,
    component: str,
    scenarios: list,
    plot_legend: bool = False,
) -> None:
    time = np.arange(2006, 2301)
    scenario_colors = {
        scenarios[0]: "#800000",
        scenarios[1]: "#ff0000",
        scenarios[2]: "#fc7b03",
        scenarios[3]: "#d3a640",
        scenarios[4]: "#098740",
        scenarios[5]: "#0080d0",
        # scenarios[6]: '#100060',
    }
    for scenario in reversed(scenarios):
        ax.fill_between(
            time,
            component_dict[scenario][component][1],
            component_dict[scenario][component][5],
            color=scenario_colors[scenario],
            edgecolor="none",
            alpha=0.3,
        )
        ax.plot(
            time,
            component_dict[scenario][component][3],
            label=f"{scenario}",
            color=scenario_colors[scenario],
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("SLE (m)")
    ax.set_title(component)
    if plot_legend:
        ax.legend(frameon=False, loc="upper left")


def main(args):
    percentiles = [0, 5, 17, 50, 83, 95, 100]
    if args.emissions_file:
        scen_df = pd.read_csv(args.emissions_file)
        scenarios = scen_df["scenario"].unique().tolist()
    else:
        # Defaulting to an SSP list to avoid UnboundLocalError
        scenarios = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"]
        # scenarios = [
        #     "Very Low - SSP1 (Marker)",
        #     "Low-to-Negative - SSP2 (Marker)",
        #     "Low - SSP2 (Marker)",
        #     "Medium-to-Low - SSP2 (Marker)",
        #     "Medium - SSP2 (Marker)",
        #     "High-to-Low - SSP5 (Marker)",
        #     "High - SSP3 (Marker)",
        # ]

    console.log(f"Using scenarios: {scenarios}")

    if "ssp" in scenarios[0].lower():
        emissions_path = args.cumulative_emissions_file
        with open(emissions_path) as f:
            cumulative_emissions = json.load(f)

    baseline_start = 1995
    baseline_end = 2014

    # Updated keys to align with the grouped components
    components = {}
    for scenario in scenarios:
        components[scenario] = {
            "gmslr": [],
            "expansion": [],
            "antarctica": [],
            "greenland": [],
            "glacier": [],
            "landwater": [],
        }

    if args.input.lower() == "magicc":
        tas, ohc = load_magicc_forcing(
            args.input_path, scenarios, baseline_start, baseline_end
        )
    elif args.input.lower() == "fair":
        tas, ohc = load_fair_forcing(
            args.input_path, scenarios, baseline_start, baseline_end
        )
    elif args.input.lower() == "run_fair":
        tas, ohc = run_fair(baseline_start, baseline_end, scenarios, args)
    else:
        raise ValueError("Input source not recognised.")

    n_iterations = 1000
    rng = np.random.default_rng()
    random_indices = rng.integers(0, high=tas.shape[1], size=n_iterations)

    tas_matrix = tas[:, random_indices, :]  # Shape: (n_scenarios, 1000, 295)
    ohc_matrix = ohc[:, random_indices, :]  # Shape: (n_scenarios, 1000, 295)

    wais_params_path = Path("components") / "aux_data" / "wais_params_expanded.nc"
    eais_params_path = Path("components") / "aux_data" / "eais_params_expanded.nc"
    pen_params_path = Path("components") / "aux_data" / "pen_params_expanded.nc"
    sampled_components = {}
    for idx, scenario in track(
        enumerate(scenarios),
        total=len(scenarios),
        description="Projecting scenarios...",
    ):
        tas_scen = tas_matrix[idx, :, :]
        ohc_scen = ohc_matrix[idx, :, :]

        slr_components = {
            "expansion": ThermalExpansion(OHC_change=ohc_scen),
            "greenland": GreenlandAR6(),
            "landwater": LandwaterAR6(),
            "wais": AntarcticaISMIP6(params_path=wais_params_path),
            "eais": AntarcticaISMIP6(params_path=eais_params_path),
            "pen": AntarcticaISMIP6(params_path=pen_params_path),
            "glacier": Glacier(),
        }

        global_model = Global(components=slr_components, end_yr=2301, nm=1)
        projections = global_model.run(
            scenario=scenario,
            T_change=tas_scen,
            member_seed=42,  # Only need one seed for the whole matrix operation
        )
        global_model.sum_components(projections)

        projections["antarctica"] = (
            projections.pop("wais") + projections.pop("eais") + projections.pop("pen")
        )

        # Process the 1000x295 outputs for saving
        sampled_components[scenario] = process_global_ensemble(
            projections, percentiles, scenario
        )

    output_dir = Path(args.output_dir) / args.output_filename
    save_to_netcdf(sampled_components, output_dir)

    fig = plt.figure(figsize=(16, 8), layout="constrained")
    ax = fig.add_subplot(231)
    plot_component(ax, sampled_components, "gmslr", scenarios, plot_legend=True)
    ax = fig.add_subplot(232)
    plot_component(ax, sampled_components, "expansion", scenarios)
    ax = fig.add_subplot(233)
    plot_component(ax, sampled_components, "glacier", scenarios)
    ax = fig.add_subplot(234)
    plot_component(ax, sampled_components, "antarctica", scenarios)
    ax = fig.add_subplot(235)
    plot_component(ax, sampled_components, "greenland", scenarios)
    ax = fig.add_subplot(236)
    plot_component(ax, sampled_components, "landwater", scenarios)

    fig.savefig(
        f"{args.output_dir}{args.output_filename.replace('.nc', '_components.png')}",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    p.add_argument(
        "--input",
        default="run_fair",
        required=False,
        help="Input climate forcing source (default: FaIR)",
        type=str,
    )
    p.add_argument(
        "--input_path", required=False, help="Path to input climate forcing", type=str
    )
    p.add_argument(
        "--emissions_file",
        required=False,
        help="Path to CSV file containing emissions scenarios (optional)",
        type=str,
    )
    p.add_argument(
        "--forcing_file",
        required=False,
        help="Path to CSV file containing climate forcing time series (optional)",
        type=str,
    )
    p.add_argument(
        "--output_dir",
        default="",
        help="Directory to save outputs (default: current directory)",
        type=str,
    )
    p.add_argument(
        "--output_filename",
        default="gmslr_projections.nc",
        help="Filename for output NetCDF (default: gmslr_projections.nc)",
        type=str,
    )
    p.add_argument(
        "--cumulative_emissions_file",
        default="cumulative_cmip6_emissions.json",
        help="Path to JSON file containing cumulative emissions for SSP scenarios (default: cumulative_cmip6_emissions.json)",
        type=str,
    )
    main(p.parse_args())
