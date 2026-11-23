.. title:: ProFSea

.. grid:: 1 1 2 2
   :margin: 4 4 0 0
   :padding: 0

   .. grid-item::
      :columns: 12 12 8 8

      .. rst-class:: display-2 font-weight-bold

      ProFSea

      .. rst-class:: display-8 font-weight-light
      
      (pronouced "pro-fe-see")

      .. rst-class:: lead

      **The Met Office's modular, accessible, and *fast* sea-level rise simulator based on statistical emulations of physical modelling experiments and lines of evidence from the IPCC and across the literature.**

      .. container:: d-flex gap-3 pt-3
         
         .. button-ref:: user_guide
            :ref-type: doc
            :color: primary
            :shadow:
            :class: font-weight-bold
            
            Get Started
            
         .. button-link:: https://github.com/MetOffice/ProFSea-tool
            :color: secondary
            :shadow:
            :outline:
            :class: font-weight-bold
            
            View on GitHub

   .. grid-item::
      :columns: 12 12 4 4

      .. image:: /_static/logo.png
         :alt: ProFSea Logo
         :class: align-center transparent-logo
         :width: 100%


----

ProFSea makes complex sea-level rise simulations easy and efficient. All you need is global mean surface temperature and ocean heat content forcing anomalies, using any baseline period, and you're good to go.

We simulate sea-level change contributions from:

- Antartic Ice Sheet 🇦🇶
- Greenland Ice Sheet 🧊
- Glacier melt 🏔️
- Thermal expansion 🌡️
- Landwater ⛰️

Quick Install
-------------

ProFSea is available as a Python package. To install it:

.. code-block:: bash

   pip install profsea

.. note::
   ProFSea relies on standard scientific libraries including ``numpy``, ``xarray``, and ``dask``. Check the :doc:`user_guide` for detailed dependency requirements.

Quickstart
----------

Build a model from the components you want, run it on your forcing, then map the global
ensemble onto sites (or a grid, with ``Spatial``). The regional data are downloaded
automatically the first time a regional component is created.

.. code-block:: python

   import numpy as np

   from profsea.components.core import Global
   from profsea.components.core.local_model import Local
   from profsea.components.global_ import AntarcticaISMIP6
   from profsea.components.spatial import Fingerprint

   # Forcing: one warming trajectory (K), 2006-2300, as an anomaly relative
   # to the 1995-2014 baseline
   years = np.arange(2006, 2301)
   t_change = np.minimum(0.03 * (years - 2005), 3.0)[None, :]

   # 1. Global mean sea-level rise from the West and East Antarctic ice sheets
   global_model = Global(
       components={
           "wais": AntarcticaISMIP6(region="wais"),
           "eais": AntarcticaISMIP6(region="eais"),
       },
       start_yr=2006,
       end_yr=2301,
       baseline_start=1995,
       baseline_end=2014,
       num_members=1000,
   )
   projections = global_model.run(scenario="ramp", T_change=t_change, member_seed=42)
   antarctic_slr = global_model.sum_components(projections)  # metres

   # 2. Relative sea-level change at a site, from the same global ensemble
   local_model = Local(
       components={
           "wais": Fingerprint(projections["wais"], fingerprint_component="wais"),
           "eais": Fingerprint(projections["eais"], fingerprint_component="eais"),
       },
       locations={"Aberdeen": (57.144, -2.080)},
   )
   local = local_model.run(member_seed=42)
   local_antarctic_contribution = local_model.sum_components(local)  # (percentile, time, site), metres
   print(local_antarctic_contribution.sel(time=2100, site="Aberdeen").values)

See the :doc:`user_guide` and tutorial notebooks for more.

----

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: User Guide
       :link: user_guide
       :link-type: doc
       :class-card: sd-rounded-3
       
       Learn how to use ProFSea and run the tutorial notebook.

   .. grid-item-card:: API Reference
       :link: api_reference
       :link-type: doc
       :class-card: sd-rounded-3
       
       Detailed descriptions of ProFSea functionality by module.

   .. grid-item-card:: Development & Docs
       :link: documentation
       :link-type: doc
       :class-card: sd-rounded-3
       
       Guidelines for contributing code and updating this documentation.

   .. grid-item-card:: References
       :link: references
       :link-type: doc
       :class-card: sd-rounded-3
       
       Academic and software references for the project.

Getting in Touch
----------------

Whether you need help getting started with ProFSea, found a bug, want the emulator to be more awesome, or just want to chat about sea-level modelling, you have a few options:

* **Open an issue** on the `GitHub repository <https://github.com/MetOffice/ProFSea-tool/issues>`_ if you find a bug, need a missing feature, or spot a typo in this documentation 👀.
* **Start a discussion** on GitHub for general questions about emulations, statistical methods, or setting up a new experiment.

Citing ProFSea
--------------

If you use ProFSea for your research, teaching, or analysis, please credit the project by citing the relevant academic references. See the :doc:`references` page for the full bibliography, methodology papers, and software DOIs.

.. toctree::
   :caption: User Guide
   :maxdepth: 2
   :hidden:

   user_guide

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:
   
   documentation

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   api_reference
   references