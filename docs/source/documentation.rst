How to update the documentation
========================================

This is the page describing how to update the sphinx documentation.

If you want to update the documentation, follow these steps to ensure that your changes are tested locally before contributing:

1. Initial Steps
-----------------------------------
First of all you need to open an issue, fork and clone the repository. For detailed instructions, refer to the following sections in the :ref:`Development Guide <development_guide>` page:

- :ref:`Open an Issue <open-an-issue>`
- :ref:`Fork the Repository <fork-the-repository>`
- :ref:`Clone Your Fork <clone-your-fork>`

2. Set Up the Environment
-------------------------
Ensure you have the required dependencies installed to build the documentation.

2.1 Create a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are using Conda, create the environment from the ``docs-environment.yml`` file:

.. code-block:: bash

   conda env create -f docs-environment.yml
   conda activate profsea-sphinx-docs

2.2 Install Dependencies with Pip (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are not using Conda, install the dependencies with ``pip``:

.. code-block:: bash

   pip install sphinx pydata-sphinx-theme

3. Make Changes to the Documentation
-------------------------------------
Edit or add ``.rst`` files in the ``docs/source`` directory. For example:

- Update ``index.rst`` to include new pages.
- Modify existing ``.rst`` files to update content.
- Add new ``.rst`` files for additional sections.

4. Build the Documentation Locally
-----------------------------------
Before committing your changes, build the documentation locally to ensure there are no errors:

.. code-block:: bash

   cd docs
   make html

This will generate the HTML files in the ``docs/build/html`` directory. Open the ``index.html`` file in your browser to preview the changes:

.. code-block:: bash

   xdg-open build/html/index.html

5. Test the Documentation
--------------------------
- Verify that all links work correctly.
- Ensure the side menu and table of contents are updated.
- Check for any warnings or errors in the terminal output during the build process.

6. Commit Changes and Create a Pull Request
---------------------------------------------
Once you are satisfied with your changes, commit and push them to your fork. For detailed instructions,
refer to the following sections in the :ref:`Development Guide <development_guide>` page:
- :ref:`Commit Changes to Your Fork <commit-changes-to-your-fork>`
- :ref:`Create a Pull Request <create-a-pull-request>`

7. Verify Deployment
---------------------
After your pull request is reviewed and merged, the GitHub Actions workflow will automatically build and deploy the updated documentation. Verify that the changes are live on GitHub Pages:

1. Go to the **Actions** tab in the main repository.
2. Check the logs for the ``deploy-docs.yml`` workflow to ensure it completed successfully.
3. Visit the GitHub Pages URL to confirm the updates.

Notes
-----
