.. title:: ProFSea

.. grid:: 1 1 2 2
   :margin: 4 4 0 0
   :padding: 0

   .. grid-item::
      :columns: 12 12 8 8

      .. rst-class:: display-2 font-weight-bold

      ProFSea

      .. rst-class:: lead

      **🌊 A modular, easy-to-setup, and self-contained sea-level rise simulator based on statistical emulations of physical modelling experiments and lines of evidence from the IPCC.**

      .. container:: d-flex gap-3 pt-3
         
         .. button-ref:: user_guide
            :ref-type: doc
            :color: primary
            :shadow:
            :class: font-weight-bold
            
            Get Started
            
         .. button-link:: [https://github.com/MetOffice/ProFSea-tool](https://github.com/MetOffice/ProFSea-tool)
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

ProFSea makes complex sea-level rise simulations accessible and fast. All you need is global mean surface temperature and ocean heat content forcing anomalies, using any baseline period, and you're good to go. ProFSea development is supported by the MetOffice.

Quick Install
-------------

ProFSea is available as a Python package. To install it:

.. code-block:: bash

   pip install profsea

.. note::
   ProFSea relies on standard scientific libraries including ``numpy``, ``xarray``, and ``dask``. Check the :doc:`user_guide` for detailed dependency requirements.

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