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

	# Initialise the Global model with the constituent components
	global_model = Global(components=global_components, end_yr=2301, num_members=1000)  # number of members you want per climate trajectory

	# Run the model forward!
	projections = global_model.run(
		T_change=t_change, scenario="stabilisation", member_seed=42
	)

	# Sum the results for the global total 
	global_model.sum_components(projections)
	gmslr = global_model.results["total_gmslr"]


Tutorial Notebook
-----------------

The worked tutorial notebook is available below:

.. toctree::
	:maxdepth: 1

	worked_example
	development_guide
	documentation
