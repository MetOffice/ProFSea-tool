import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

from profsea.components.core.global_model import Global
from profsea.components.global_.greenland import GreenlandAR6
from profsea.components.global_.expansion import ThermalExpansion
from profsea.components.global_.antarctica import AntarcticaDynAR5, AntarcticaSMBAR5
from profsea.components.core.spatial_model import Spatial
from profsea.components.spatial import SterodynamicCMIP6, Fingerprint

### Global projections first ###
slr_components = {
    "expansion": ThermalExpansion(
        OHC_change=np.linspace(1, 5, 295).reshape(1, -1) * 1e24
    ),
    "greenland": GreenlandAR6(),
}

# Pass to the global model
model = Global(components=slr_components, end_yr=2301)
projections = model.run(
    scenario="test",
    T_change=np.linspace(1, 5, 295).reshape(1, -1),
    member_seed=42,
)

### Now spatial projections ###
spatial_components = {
    "sterodynamic": SterodynamicCMIP6(
        projections["expansion"],
        patterns_dir="/Users/gregorymunday/Documents/Papers/ProFSea/ProFSea-tool/data/cmip6",
    ),
    "greenland": Fingerprint(
        projections["greenland"],
        fingerprint_paths="/Users/gregorymunday/Documents/Papers/ProFSea/ProFSea-tool/data/grd_fingerprints/greenland_ar6.nc",
        scaling_factor=1e3,  # convert from m to mm
    ),
}

# Pass to the spatial model
model = Spatial(components=spatial_components)
model.run(scenario="test", member_seed=42)

model.sum_components(model.results)
model.save_components(model.results, scenario_name="test", output_format="zarr")


### Plot example ###
fig = plt.figure(figsize=(10, 5))

total_rsl = model.results["total_rsl"][3, -1, :, :]
vmax = np.nanmax(np.abs(total_rsl))
vmin = -vmax

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.pcolormesh(
    model.grid_lons,
    model.grid_lats,
    model.results["total_rsl"][3, -1, :, :],
    transform=ccrs.PlateCarree(),
    cmap="PuOr_r",
    vmin=vmin,
    vmax=vmax,
)
ax.coastlines()
fig.colorbar(mappable=ax.collections[0], label="Sterodynamic SLR contribution (mm/yr)")
plt.show()
