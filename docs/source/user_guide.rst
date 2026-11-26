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


Regional Data
-------------

The regional components (``Fingerprint``, ``SterodynamicCMIP6`` and ``GIA``) use
patterns that are too large to ship with the package. They are downloaded from
`Zenodo <https://doi.org/10.5281/zenodo.20427061>`__ (about 50 MB) the first time a
regional component or model is created, and reused afterwards. They are stored in your
user cache directory (for example ``~/.cache/profsea`` on Linux or
``~/Library/Caches/profsea`` on macOS). To keep them elsewhere, for instance on a
shared drive, set the ``PROFSEA_DATA_DIR`` environment variable to a folder of your
choice before running ProFSea. The :doc:`development_guide` describes how a source
checkout finds them.

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
