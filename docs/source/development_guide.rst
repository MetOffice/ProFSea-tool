.. _development_guide:

Development Guide
======================

This is the Development Guide page.

.. _open-an-issue:

Open an Issue
-------------
Before forking the repository, open an issue in the main repository to describe the changes you plan to make. This helps maintainers track contributions and provide feedback early. Check with the repository owners what is the name of the development branch you need to use. In this guide, the name used for the development branch is generically indicated as "development_branch_name"

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
Clone your forked repository and switch to the `development_branch_name` branch:

.. code-block:: bash

   git clone git@github.com:<your-username>/ProFSea-tool.git
   cd ProFSea-tool
   git checkout <development_branch_name>

Replace ``<your-username>`` with your GitHub username.

Replace ``<development_branch_name>`` with the development branch name provided by the repository owners

.. _commit-changes-to-your-fork:

Commit Changes to Your Fork
---------------------------
Once you are satisfied with your changes, commit and push them to your fork:

.. code-block:: bash

   git add .
   git commit -m "Adding feature: <brief description of changes>"
   git push origin development_branch_name

Replace ``<brief description of changes>`` with a short summary of your updates.

.. _create-a-pull-request:

Create a Pull Request
---------------------
To contribute your changes to the main repository:

1. Go to your forked repository on GitHub.
2. Click the **Pull Request** button.
3. Select the ``development_branch_name`` branch of the main repository as the base branch.
4. Select the ``development_branch_name`` branch of your fork as the compare branch.
5. Add a title and description for your pull request.
6. Click **Create Pull Request**.
7. Assign reviewers to the pull request
