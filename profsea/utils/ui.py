from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Central console for package-generated UI elements (not logs)
ui_console = Console()


def print_global_preflight(model, scenario: str) -> None:
    """Renders the pre-flight summary for the Global model."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan", justify="right")
    table.add_column("Value", style="magenta")

    components_list = ", ".join(model.components.keys())
    ensemble_size = model.nt * model.num_members

    if model.output_percentiles is not None:
        output_str = f"Percentiles: {model.output_percentiles}"
    else:
        output_str = f"Full Distribution ({ensemble_size} members)"

    table.add_row("Scenario", scenario)
    table.add_row("Components", components_list)
    table.add_row("Timeframe", f"{model.endofhistory} -> {model.end_yr}")
    table.add_row(
        "Ensemble Size",
        f"{model.nt} inputs × {model.num_members} draws (= {ensemble_size})",
    )
    table.add_row("Output", output_str)
    table.add_row(
        "Compute Engine", "Parallel Threading" if model.parallel else "Sequential"
    )

    panel = Panel(
        table,
        title="[bold green]ProFSea Global Run Configuration",
        expand=False,
        border_style="green",
    )
    ui_console.print(panel)
    ui_console.print()


def print_spatial_preflight(model) -> None:
    """Renders the pre-flight summary for the Spatial model."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan", justify="right")
    table.add_column("Value", style="magenta")

    components_list = ", ".join(model.components.keys())

    # Calculate the mathematical grid size: $N_{lat} \times N_{lon}$
    n_lat = len(model.grid_lats)
    n_lon = len(model.grid_lons)
    grid_str = f"{n_lat} × {n_lon} cells"

    if model.output_percentiles is not None:
        output_str = f"Percentiles: {model.output_percentiles}"
        members = len(model.output_percentiles)
    else:
        output_str = f"Full Distribution ({model.num_members} members)"
        members = model.num_members

    # Memory estimation
    bytes_per_element = 8
    future_size_gb = (members * model.n_years * n_lat * n_lon * bytes_per_element) / 1e9

    table.add_row("Components", components_list)
    table.add_row("Timeframe", f"{model.start_year} -> {model.end_year}")
    table.add_row("Baseline", f"{model.baseline_yrs[0]} - {model.baseline_yrs[1]}")
    table.add_row("Grid Resolution", grid_str)
    table.add_row("Output", output_str)

    # Color-code the memory warning
    mem_style = "bold red" if future_size_gb > 20 else "magenta"
    table.add_row(
        "Est. Output Size", f"[{mem_style}]~{future_size_gb:.2f} GB[/{mem_style}]"
    )

    panel = Panel(
        table,
        title="[bold blue]ProFSea Spatial Run Configuration",
        expand=False,
        border_style="blue",
    )
    ui_console.print(panel)
    ui_console.print()
