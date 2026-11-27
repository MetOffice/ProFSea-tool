User Guide
==========

ProFSea is designed as a modular simulator, where you can build their own model with whatever sea level components you want to run with. The original ProFSea technical document can be found `here <proflink_>`__. The Monte Carlo sampling method used in ProFSea v2.0.0 is used here with some modifications depending on the component. Further technical and methodological about the latest ProFSea version will be submitted as a journal paper soon.

.. _proflink: https://zenodo.org/records/10255468

Detailed examples on how to use ProFSea can be found in the tutorial notebooks, but here's a quick example to get started.

**Running a global sea level projection**:

.. code-block:: python

	# Import all the components you want to use in your model
	from profsea.components.core.global_model import Global
	from profsea.components.global_ import (
		AntarcticaISMIP6,
		Glacier,
		GreenlandAR6,
		LandwaterAR6,
		ThermalExpansion,
	)

	global_components = {
		"landwater": LandwaterAR6(),
		"greenland": GreenlandAR6(),
		"expansion": ThermalExpansion(ohc_change),  # only the thermal expansion needs OHC change, so we'll pass it in here
		"wais": AntarcticaISMIP6(region="wais"),
		"eais": AntarcticaISMIP6(region="eais"),
		"peninsula": AntarcticaISMIP6(region="peninsula"),
		"glacier": Glacier(),
	}

	# Initialise the Global model with the constituent components.
	# t_change and ohc_change must be anomalies relative to the same
	# baseline period passed here.
	global_model = Global(
		components=global_components,
		start_yr=2006,
		end_yr=2301,
		baseline_start=1995,
		baseline_end=2014,
		num_members=1000,  # number of members you want per climate trajectory
	)

	# Run the model forward!
	projections = global_model.run(
		T_change=t_change, scenario="stabilisation", member_seed=42
	)

	# Sum the results for the global total 
	global_model.sum_components(projections)
	gmslr = global_model.results["total_gmslr"]


Spatial and local ensembles
---------------------------

``Spatial`` and ``Local`` draw ``num_spatial_members`` (default 1000) members
from the full global ensemble, so run ``Global`` with ``output_percentiles=None``.
Every climate trajectory gets an equal share of the members, and every component
uses the same members, so ``total_rsl`` (added when ``compute_total=True``, the
default) is summed member by member. Each member also draws its spatial
patterns (GRD fingerprint, CMIP sterodynamic pattern and GIA model) at random,
independently of its global member, and keeps the same GRD model for every
fingerprint component. This follows the local-projection method of Palmer et
al. (2020, *Earth's Future*, doi:10.1029/2019EF001413). Set
``sample_spatial=False`` on a component to use the mean of its patterns
instead.

Percentiles (``output_percentiles``, default ``[5, 17, 50, 83, 95]``) are taken
over the members at every grid point or site and year. With
``output_percentiles=None`` or ``[]`` the outputs keep every member, with
dimensions ``(member, time, lat, lon)`` or ``(member, time, site)``; the
``climate_member`` and ``process_member`` coordinates record which global member
each one came from. Full-member output is large on a grid (about 76 GB per
variable for 1000 members on the 1° grid), so prefer percentiles, a coarser grid
or ``Local`` for site time series. ``chunk_size_mb`` (default 64) sets the
approximate memory of one block of the calculation.

At a coastal site, a model whose four surrounding grid cells are all land has
no value there, and members that drew it are left out of that site's
percentiles. Check how many members remain (``output_percentiles=None``) for
sites close to the coast.

Tutorial Notebook
-----------------

The worked tutorial notebook is available below:

.. toctree::
	:maxdepth: 1

	tutorial_notebooks/worked_example
	tutorial_notebooks/zarr_output_files
	tutorial_notebooks/observations_and_baselines
	development_guide
	documentation
