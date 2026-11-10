

from profsea.components.core.local_model import Local

from profsea.components.core import Spatial
from profsea.components.spatial import (
    SterodynamicCMIP6,
    Fingerprint,
    GIA,
)

import xarray as xr
import pdb



def main():

    locations = {
        "Aberdeen": (57.144, -2.080),
        "Holyhead": (53.314, -4.620),
        "Llandudno": (53.330, -3.830),
        "Land point": (-34.6037, -58.3816),
    }

    local_components = {
        "sterodynamic": SterodynamicCMIP6(
            projections["expansion"],  # passing in the global thermal expansion projection
        ),
        "greenland": Fingerprint(
            projections["greenland"], 
            fingerprint_component="greenland"
        ),
        "landwater": Fingerprint(
            projections["landwater"],
            fingerprint_component="landwater",
        ),
        "wais": Fingerprint(
            projections["wais"],
            fingerprint_component="wais",
        ),
        "eais": Fingerprint(
            projections["eais"],
            fingerprint_component="eais",
        ),
        "glacier": Fingerprint(
            projections["glacier"],
            fingerprint_component="glacier",
        ),
        "gia": GIA(
            sample_spatial=False,
        ),
    }

    local_model = Local(components=local_components, locations=locations)
    local_timeseries = local_model.run(member_seed=42)
    total_rsl = local_model.sum_components(local_model.results)




    

    rcp85_proj = xr.open_zarr("rcp85_projection.zarr")
    print(rcp85_proj)
    print(rcp85_proj.data_vars)

    print(rcp85_proj["total_rsl"])

    print(rcp85_proj["total_rsl"][2,:,50,50].values)

    rcp85_proj["total_rsl"].sel(percentile=50)

    print(rcp85_proj["total_rsl"].sel(percentile=50, lon=300, lat=50, method='nearest'))

    rcp85_proj["total_rsl"].sel(percentile=50, lon=300, lat=50, method='nearest').values

    rcp85_proj["total_rsl"].sel(percentile=50, lon=300, lat=50, method='nearest').dims

    pdb.set_trace()


if __name__ == '__main__':
    main()
