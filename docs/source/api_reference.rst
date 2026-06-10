API Reference
=============

This page exposes selected Python API docs generated from in-code docstrings.

Core Models
-----------

.. automodule:: profsea.components.core.global_model
   :members: Global
   :undoc-members:
   :show-inheritance:

.. automodule:: profsea.components.core.spatial_model
   :members: Spatial, fetch_zenodo_fingerprints
   :undoc-members:
   :show-inheritance:

.. automodule:: profsea.components.core.state
   :members: ClimateState, SpatialState

.. automodule:: profsea.components.core.time_projection
   :members: time_projection

Global Components
-----------------

.. automodule:: profsea.components.global_.antarctica
   :members: AntarcticaISMIP6, AntarcticaDynAR5, AntarcticaSMBAR5

.. automodule:: profsea.components.global_.greenland
   :members: GreenlandAR6, GreenlandSMBAR5, GreenlandDynAR5, load_greenland_calibration

.. automodule:: profsea.components.global_.glacier
   :members: Glacier

.. automodule:: profsea.components.global_.expansion
   :members: ThermalExpansion

.. automodule:: profsea.components.global_.landwater
   :members: LandwaterAR6, LandwaterAR5, load_landwater_projection

Spatial Components
------------------

.. automodule:: profsea.components.spatial.fingerprint
   :members: Fingerprint

.. automodule:: profsea.components.spatial.sterodynamic
   :members: SterodynamicCMIP6, SterodynamicCMIP5

.. automodule:: profsea.components.spatial.gia
   :members: GIA
