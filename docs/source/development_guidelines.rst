.. _development_guidelines:

Development Guidelines
======================

This is the Development Guidelines page.

.. _open-an-issue:

Open an Issue
-------------
Before forking the repository, open an issue in the main repository to describe the changes you plan to make. This helps maintainers track contributions and provide feedback early.

.. _fork-the-repository:

Fork the Repository
-------------------
First, create a fork of the repository in your GitHub account:

1. Go to the repository page on GitHub.
2. Click the **Fork** button in the top-right corner.
3. Select your GitHub account as the destination for the fork.

.. _clone-your-fork:

Clone Your Fork
---------------
Clone your forked repository and switch to the `profsea-climate` branch:

.. code-block:: bash

   git clone git@github.com:<your-username>/ProFSea-tool.git
   cd ProFSea-tool
   git checkout profsea-climate

Replace ``<your-username>`` with your GitHub username.

.. _commit-changes-to-your-fork:

Commit Changes to Your Fork
---------------------------
Once you are satisfied with your changes, commit and push them to your fork:

.. code-block:: bash

   git add .
   git commit -m "Adding feature: <brief description of changes>"
   git push origin profsea-climate

Replace ``<brief description of changes>`` with a short summary of your updates.

.. _create-a-pull-request:

Create a Pull Request
---------------------
To contribute your changes to the main repository:

1. Go to your forked repository on GitHub.
2. Click the **Pull Request** button.
3. Select the ``profsea-climate`` branch of the main repository as the base branch.
4. Select the ``profsea-climate`` branch of your fork as the compare branch.
5. Add a title and description for your pull request.
6. Click **Create Pull Request**.
