import numpy as np
import matplotlib.pyplot as plt

from profsea.components.core.global_model import Global
from profsea.components.global_.greenland import GreenlandAR6
from profsea.components.global_.expansion import ThermalExpansion
from profsea.components.global_.antarctica import AntarcticaDynAR5, AntarcticaSMBAR5
from profsea.components.core.spatial_model import Spatial
from profsea.components.spatial.sterodynamic import SterodynamicCMIP6


slr_components = {
    "expansion": ThermalExpansion(
        OHC_change=np.linspace(1, 5, 295).reshape(1, -1) * 1e24
    ),
}

model = Global(components=slr_components, end_yr=2301)

projections = model.run(
    scenario="test",
    T_change=np.linspace(1, 5, 295).reshape(1, -1),
    member_seed=42,
)


spatial_components = {
    "sterodynamic": SterodynamicCMIP6(
        projections["expansion"],
        patterns_dir="/Users/gregorymunday/Documents/Papers/ProFSea/ProFSea-tool/data/cmip6",
    ),
}

# Now pass to the spatial model
model = Spatial(components=spatial_components)
model.run(scenario="test", member_seed=42)

model.sum_components(model.results)
model.save_components(
    model.results, 
    scenario_name="test", 
    output_format="zarr"
)

plt.pcolormesh(model.grid_lons, model.grid_lats, model.results["sterodynamic"][3, -1, :, :])
plt.colorbar(label="Sterodynamic SLR contribution (mm/yr)")
plt.show()
