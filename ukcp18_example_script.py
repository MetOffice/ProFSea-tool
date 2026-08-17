import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from profsea.components.core.global_model import Global
from profsea.components.core.spatial_model import Spatial
from profsea.components.global_ import (
    AntarcticaDynAR5,
    AntarcticaSMBAR5,
    Glacier,
    GreenlandDynAR5,
    GreenlandSMBAR5,
    LandwaterAR5,
    ThermalExpansion,
)
from profsea.components.spatial import GIA, Fingerprint, SterodynamicCMIP6

### Global projections first ###
global_components = {
    "expansion": ThermalExpansion(
        OHC_change=np.linspace(1, 5, 295).reshape(1, -1) * 1e24
    ),
    "greenland_dyn": GreenlandDynAR5(),
    "greenland_smb": GreenlandSMBAR5(),
    "landwater": LandwaterAR5(),
    "antarctica_dyn": AntarcticaDynAR5(),
    "antarctica_smb": AntarcticaSMBAR5(),
    "glacier": Glacier(),
}

# Pass to the global model
global_model = Global(components=global_components, end_yr=2301)
projections = global_model.run(
    scenario="rcp85",
    T_change=np.linspace(1, 5, 295).reshape(1, -1),
    member_seed=42,
)
global_model.sum_components(projections)
gmslr = global_model.results["total_gmslr"]
global_model.save_components(
    global_model.results,
    scenario_name="rcp85",
    output_format="zarr",
    output_prefix="global_",
)

### Now spatial projections ###
spatial_components = {
    "sterodynamic": SterodynamicCMIP6(
        projections["expansion"],
    ),
    "greenland_dyn": Fingerprint(
        projections["greenland_dyn"], fingerprint_component="greendyn"
    ),
    "greenland_smb": Fingerprint(
        projections["greenland_smb"],
        fingerprint_component="greensmb",
    ),
    "landwater": Fingerprint(
        projections["landwater"],
        fingerprint_component="landwater",
    ),
    "antarctica_dyn": Fingerprint(
        projections["antarctica_dyn"],
        fingerprint_component="antdyn",
    ),
    "antarctica_smb": Fingerprint(
        projections["antarctica_smb"],
        fingerprint_component="antsmb",
    ),
    "glacier": Fingerprint(
        projections["glacier"],
        fingerprint_component="glacier",
    ),
    "gia": GIA(
        sample_spatial=False,
    ),
}

# Pass to the spatial model
spatial_model = Spatial(components=spatial_components)
spatial_model.run(member_seed=42)

spatial_model.sum_components(spatial_model.results)
spatial_model.save_components(
    spatial_model.results, scenario_name="rcp85", output_format="zarr"
)


### Plot example ###
fig = plt.figure(figsize=(10, 4), layout="constrained")

total_rsl = spatial_model.results["total_rsl"].isel(percentile=2, time=-1)
vmax = np.nanmax(np.abs(total_rsl))
vmin = -vmax

ax = fig.add_subplot(121)

# Global projections
yrs = np.arange(2006, 2301)
ax.plot(
    yrs,
    np.median(gmslr, axis=(0, 1)),
    label="Global Projection",
    color="royalblue",
)
# fill between 1 and 4 members
ax.fill_between(
    yrs,
    np.percentile(gmslr, 17, axis=(0, 1)),
    np.percentile(gmslr, 83, axis=(0, 1)),
    alpha=0.3,
    color="skyblue",
)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Global Mean Sea Level Rise (m)")

ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
ax.set_title("50th Percentile, Year 2300")
ax.pcolormesh(
    spatial_model.grid_lons,
    spatial_model.grid_lats,
    total_rsl,
    transform=ccrs.PlateCarree(),
    cmap="PuOr_r",
    vmin=vmin,
    vmax=vmax,
)
ax.coastlines()
fig.colorbar(
    mappable=ax.collections[0],
    label="Relative Sea Level (m)",
    orientation="horizontal",
    pad=0.02,
)
plt.show()
