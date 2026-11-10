import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from profsea.components.core.global_model import Global
from profsea.components.core.spatial_model import Spatial
from profsea.components.core.local_model import Local

from profsea.components.global_ import (
    AntarcticaDynAR5,
    AntarcticaSMBAR5,
    Glacier,
    GreenlandDynAR5,
    GreenlandSMBAR5,
    LandwaterAR5,
    ThermalExpansion,
    #ThermalExpansion_via_zostoga
)
from profsea.components.spatial import GIA, Fingerprint, SterodynamicCMIP6

# other python modules
from profsea.utils import sample_members_2D
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import yaml
import json

def main():


#    ### Global projections first ###
#    global_components = {
#        "expansion": ThermalExpansion(
#            OHC_change=np.linspace(1, 5, 295).reshape(1, -1) * 1e24
#        ),
#        "greenland_dyn": GreenlandDynAR5(),
#        "greenland_smb": GreenlandSMBAR5(),
#        "landwater": LandwaterAR5(),
#        "antarctica_dyn": AntarcticaDynAR5(),
#        "antarctica_smb": AntarcticaSMBAR5(),
#        "glacier": Glacier(),
#    }



    # Prepare CMIP5 inputs
    cmip5_dir = '/data/users/lesley.allison/project_sea_level_copy/sea_level_projections/cmip5_timeseries/'
    scenario = 'rcp26'
    with open (cmip5_dir+scenario+'_zostoga_tas_timeseries.pickle', 'rb') as f:
        data=pickle.load(f, encoding='latin1')  # encoding to get around UnicodeDecodeError due to old pickle (python2)
    zostoga_years = data['zostoga_yrs']
    tas_years = data['tas_yrs']
    tas_series = data['tas_series']
    zostoga_series = data['zostoga_series']
    models = list(data['tas_series'].keys())
    print(models)

    
    model_pick = models[6]
    single_tas_ts = tas_series[model_pick]
    single_tas_yrs = tas_years[model_pick]
    single_zostoga_ts = zostoga_series[model_pick]
    single_zostoga_yrs = zostoga_years[model_pick]

    # Temporary workaround to get tas and zostoga the same length
    # tas starts in 2006, zostoga starts in 2007
    single_tas_ts = single_tas_ts[1:]
    single_tas_yrs = single_tas_yrs[1:]


    ### Global projections first ###
    global_components = {
        "expansion": ThermalExpansion(single_zostoga_ts),
        "greenland_dyn": GreenlandDynAR5(),
        "greenland_smb": GreenlandSMBAR5(),
        "landwater": LandwaterAR5(),
        "antarctica_dyn": AntarcticaDynAR5(),
        "antarctica_smb": AntarcticaSMBAR5(),
        "glacier": Glacier(),
    }


    
    # Pass to the global model (Initialise the global model with the consituent components)
#    global_model = Global(components=global_components, end_yr=2301)  # creating an instance of the Global class
    global_model = Global(components=global_components, end_yr=2101)  # creating an instance of the Global class
    
    # Run the global model
    projections = global_model.run(
        scenario="rcp85",
        #T_change=np.linspace(1, 5, 295).reshape(1, -1),  # reshape is making a row vector
        T_change=single_tas_ts,
        member_seed=42,
    )

    # Sum the components for the global mean total
    global_model.sum_components(projections)
    gmslr = global_model.results["total_gmslr"]

    
    pdb.set_trace()

    
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
        spatial_model.results, scenario_name="rcp85_test", output_format="zarr"
    )


    ### Plot example ###
    fig = plt.figure(figsize=(10, 4), layout="constrained")

    total_rsl = spatial_model.results["total_rsl"][2, -1, :, :]
    vmax = np.nanmax(np.abs(total_rsl))
    vmin = -vmax

    ax = fig.add_subplot(121)

    # Global projections
#    yrs = np.arange(2006, 2301)
    yrs = np.arange(2006, 2100)
    ax.plot(
        yrs,
        np.median(gmslr, axis=0),
        label="Global Projection",
        color="royalblue",
    )
    # fill between 1 and 4 members
    ax.fill_between(
        yrs,
        np.percentile(gmslr, 17, axis=0),
        np.percentile(gmslr, 83, axis=0),
        alpha=0.3,
        color="skyblue",
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Global Mean Sea Level Rise (m)")

    ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax.set_title("50th Percentile, Year 2100")
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



    locations = {
        "Aberdeen": (57.144, -2.080),
        "Holyhead": (53.314, -4.620),
        "Llandudno": (53.330, -3.830),
        "Land point": (-34.6037, -58.3816),
        "Durban": (-29.874, 31.051),
    }

    local_components = {
        "sterodynamic": SterodynamicCMIP6(
            projections["expansion"],  # passing in the global thermal expansion projection
        ),
        "greenland_smb": Fingerprint(
            projections["greenland_smb"], 
            fingerprint_component="greensmb"
        ),
        "greenland_dyn": Fingerprint(
            projections["greenland_dyn"], 
            fingerprint_component="greendyn"
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

    local_model = Local(components=local_components, locations=locations)
    local_timeseries = local_model.run(member_seed=42)
    total_rsl = local_model.sum_components(local_model.results)

    loc_ts = local_timeseries["total_rsl"].sel(site=["Aberdeen", "Holyhead", "Llandudno", "Land point", "Durban"], percentile=[17, 50, 83])
    fig = plt.figure(figsize=(6, 4), layout="constrained")
    ax = fig.add_subplot(111)
    for loc in loc_ts.site.values:
        loc_data = loc_ts.sel(site=loc)
        ax.fill_between(loc_data.time, loc_data.sel(percentile=17), loc_data.sel(percentile=83), alpha=0.3)
        ax.plot(loc_data.time, loc_data.sel(percentile=50), label=loc)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Sea Level (m)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    plt.show()

    print('end')
    pdb.set_trace()



    
if __name__ == '__main__':
    main()

