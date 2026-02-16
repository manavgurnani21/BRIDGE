# Contributing code

This guide is intended for developers who wish to contribute to our codebase. Here's how to set
up a local development environment:

## Setting up a development environment

1. Fork our [repository] on GitHub

2. Locally clone your forked repository (replace `your-username` with your GitHub username):

    ```bash
    git clone https://github.com/{your-username}/BRIDGE.git
    # or with SSH
    git clone git@github.com:{your-username}/BRIDGE.git

    cd BRIDGE
    ```

3. Add the main repository as a remote:

    ```bash
    git remote add upstream https://github.com/wangyb97/BRIDGE.git
    ```

4. Install the development dependencies and the package into a virtual environment:

    ```bash
    conda env create -f BRIDGE.yml
    conda activate BRIDGE
    ```

    Don't know how to set up a virtual environment? Check out our [installation] guide!

## Scoping changes

Before you start working on a new feature or bug fix, we recommend opening an [issue] (if one does
not already exist) to discuss the proposed changes. This will help ensure that your changes are
aligned with the project's goals and that you are not duplicating work.

We don't guarantee that all changes will be accepted, but we will do our best to provide feedback
and guidance on how to improve your contributions.

## Adding code changes

We only accept code changes that are made through pull requests. To contribute, follow these steps:

1. Create a new branch for your changes:

    ```bash
    git checkout -b my-change
    ```

2. Make your changes and commit them:

    ```bash
    git add .
    git commit -m "My change"
    ```

3. Push your changes to your fork:

    ```bash
    git push origin my-change
    ```

4. Open a pull request on the main repository. Make sure to include a detailed description of your
    changes in the body and reference any related issues.

[installation]: ../installation.md
[repository]: https://github.com/wangyb97/BRIDGE
[issue]: https://github.com/wangyb97/BRIDGE/issues/new/choose
