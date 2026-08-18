.. _updating_documentation:

Updating the Documentation
==========================

This guide explains how to appropriately update, build, and test the Sphinx documentation locally before contributing your changes to the project.

Initial Steps
-------------
Before making changes to the documentation, ensure you have set up your local workspace. For detailed instructions, refer to the following sections in the :ref:`Development Guide <development_guide>`:

* :ref:`Open an Issue <open-an-issue>`
* :ref:`Fork the Repository <fork-the-repository>`
* :ref:`Clone Your Fork <clone-your-fork>`


Set Up the Environment
----------------------
You will need a few specific dependencies installed to build the documentation locally.

**Using Conda (Recommended)**
If you are using Conda, create and activate the dedicated documentation environment from the provided YAML file:

.. code-block:: bash

    conda env create -f docs-environment.yml
    conda activate profsea-sphinx-docs

**Using Pip**
Alternatively, if you are not using Conda, you can install the required packages directly via pip:

.. code-block:: bash

    pip install sphinx pydata-sphinx-theme


Make Changes
------------
All documentation source files are written in reStructuredText (``.rst``) and are located in the ``docs/source`` directory. 

* **To add new pages:** Create a new ``.rst`` file and add its name to the `toctree` in ``index.rst``.
* **To update existing content:** Modify the relevant ``.rst`` files directly.


Build and Preview Locally
-------------------------
Before committing your changes, you must build the documentation locally to ensure there are no formatting errors or broken references.

Navigate to the ``docs`` directory and build the HTML:

.. code-block:: console

    $ cd docs
    $ make html

    Running Sphinx v7.2.6
    loading pickled environment... done
    building [mo]: targets for 0 po files that are out of date
    building [html]: targets for 1 source files that are out of date
    updating environment: 0 added, 1 changed, 0 removed
    reading sources... [100%] updating_documentation
    looking for now-outdated files... none found
    pickling environment... done
    checking consistency... done
    preparing documents... done
    copying assets... done
    writing output... [100%] updating_documentation
    build succeeded.

    The HTML pages are in _build/html.

Once the build succeeds, open the generated index file in your web browser to preview your changes. 

.. code-block:: bash

    # On macOS:
    open build/html/index.html
    
    # On Linux:
    xdg-open build/html/index.html


Test the Documentation
----------------------
While previewing the site, please verify the following:

* All internal and external hyperlinks resolve correctly.
* The side menu and table of contents accurately reflect any new headers or pages.
* The terminal output from the ``make html`` command shows no warnings or errors.


Commit Changes and Create a PR
------------------------------
Once you are satisfied with how the documentation looks and builds, commit and push the changes to your branch. For detailed instructions, refer back to the :ref:`Development Guide <development_guide>`:

* :ref:`Commit Changes to Your Fork <commit-changes-to-your-fork>`
* :ref:`Create a Pull Request <create-a-pull-request>`


Verify Deployment
-----------------
After your pull request is reviewed and merged into ``main``, a GitHub Actions workflow will automatically build and deploy the updated documentation. 

To verify that the changes are live:

1. Navigate to the **Actions** tab in the main GitHub repository.
2. Check the logs for the ``deploy-docs.yml`` workflow to ensure it completed successfully.
3. Visit the live GitHub Pages URL to confirm your updates are visible.