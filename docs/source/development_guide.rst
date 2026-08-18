.. _development-guide:

Development Guide
=================

Before making any changes, you need to set up your local workspace and install the package in "editable" mode. This allows your local changes to take effect immediately without needing to reinstall the package.

1. **Clone the Repository**
   
   Clone the main repository directly to your local machine:

   .. code-block:: bash

       git clone https://github.com/MetOffice/ProFSea-tool.git
       cd ProFSea-tool

2. **Create a Virtual Environment**

   .. code-block:: bash

       # Using Conda
       conda env create -f profsea-env.yml
       conda activate profsea

3. **Install in Editable Mode**
   
   Install the package along with its development dependencies using the ``-e`` flag:

   .. code-block:: bash

       pip install -e ".[dev]"


.. _open-an-issue:

Open an Issue
-------------
Before making any significant changes, writing new features, or submitting a Pull Request (PR), please **open an issue first**. 

Discussing your ideas with the maintainers beforehand ensures that your proposed changes align with the project's architecture and roadmap, saving you time and preventing duplicate work.


Create a Branch
---------------
Once your issue is discussed and approved, create a new branch from `main`. We use a standardized branch naming convention based on your initials and a short, descriptive hyphenated name. 

For example, if your name is John Smith and you are fixing a bug in the Greenland dynamics module, you would name your branch:

.. code-block:: bash

    git checkout -b js/fix-greenland-dynamics


Write Tests
-----------
We rely on a robust test suite to ensure the framework remains stable as it grows. All new features, bug fixes, and logic changes **must be accompanied by corresponding unit tests**. 

Please include these tests in the same commits as your code changes rather than adding them as an afterthought.


The Pre-Push Checklist
----------------------
Before you `git push` your branch and open a Pull Request, you must verify that your code passes all tests, conforms to our style guide, and doesn't break the documentation build. 

Run the following commands in your terminal from the root of the repository:

.. code-block:: console

    # Run the test suite
    $ pytest tests/
    
    =========================== test session starts ============================
    platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0
    rootdir: /path/to/ProFSea-tool
    plugins: zarr-3.2.1
    collected 119 items
    
    tests/components/test_antarctica.py ............                     [ 10%]
    tests/components/test_fingerprint.py .........                       [ 17%]
    tests/components/test_gia.py .........                               [ 25%]
    ...
    tests/utils/test_utils.py .....                                      [100%]
    
    ============================ 119 passed in 1.42s ===========================


    # Run the linter and automatically fix formatting issues
    $ ruff check . --fix
    
    Found 3 errors (3 fixed, 0 remaining).


    # Build the documentation (assuming you are using a Makefile in /docs)
    $ cd docs
    $ make html
    
    Running Sphinx v7.2.6
    loading pickled environment... done
    building [mo]: targets for 0 po files that are out of date
    building [html]: targets for 1 source files that are out of date
    updating environment: 0 added, 1 changed, 0 removed
    reading sources... [100%] development_guide
    looking for now-outdated files... none found
    pickling environment... done
    checking consistency... done
    preparing documents... done
    copying assets... done
    writing output... [100%] development_guide
    build succeeded.

    The HTML pages are in _build/html.


So without all the fluff, that's: 

.. code-block:: bash

    pytest tests/
    ruff check . --fix
    cd docs
    make html

Once all three steps complete successfully, you are ready to push your branch and open a PR!