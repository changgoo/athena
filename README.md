athena
======
<!-- Jenkins Status Badge in Markdown (with view), unprotected, flat style -->
<!-- In general, need to be on Princeton VPN, logged into Princeton CAS, with ViewStatus access to Jenkins instance to click on unprotected Build Status Badge, but server is configured to whitelist GitHub -->
<!-- [![Jenkins Build Status](https://jenkins.princeton.edu/buildStatus/icon?job=athena/PrincetonUniversity_athena_jenkins_master)](https://jenkins.princeton.edu/job/athena/job/PrincetonUniversity_athena_jenkins_master/) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4455880.svg)](https://doi.org/10.5281/zenodo.4455880) <!-- v21.0, not Concept DOI that tracks the "latest" version (erroneously sorted by DOI creation date on Zenodo). 10.5281/zenodo.4455879 -->
[![Travis CI Build Status](https://travis-ci.com/PrincetonUniversity/athena.svg?token=Ejzw3yndG1Fqub679gCB&branch=master)](https://travis-ci.com/PrincetonUniversity/athena)
[![codecov](https://codecov.io/gh/PrincetonUniversity/athena/branch/master/graph/badge.svg?token=ZzniY084kP)](https://codecov.io/gh/PrincetonUniversity/athena)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](code_of_conduct.md)

<!--[![Public GitHub  issues](https://img.shields.io/github/issues/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/issues)
[![Public GitHub pull requests](https://img.shields.io/github/issues-pr/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/pulls) -->

<p align="center">
	  <img width="345" height="345" src="https://user-images.githubusercontent.com/1410981/115276281-759d8580-a108-11eb-9fc9-833480b97f95.png">
</p>

Athena++ GRMHD code and adaptive mesh refinement (AMR) framework

Please read [our contributing guidelines](./CONTRIBUTING.md) for details on how to participate.

## Citation
To cite Athena++ in your publication, please use the following BibTeX to refer to the code's [method paper](https://ui.adsabs.harvard.edu/abs/2020ApJS..249....4S/abstract):
```
@article{Stone2020,
	doi = {10.3847/1538-4365/ab929b},
	url = {https://doi.org/10.3847%2F1538-4365%2Fab929b},
	year = 2020,
	month = jun,
	publisher = {American Astronomical Society},
	volume = {249},
	number = {1},
	pages = {4},
	author = {James M. Stone and Kengo Tomida and Christopher J. White and Kyle G. Felker},
	title = {The Athena$\mathplus$$\mathplus$ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers},
	journal = {The Astrophysical Journal Supplement Series},
}
```
Additionally, you can add a reference to `https://github.com/PrincetonUniversity/athena` in a footnote.

Finally, we have minted DOIs for each released version of Athena++ on Zenodo. This practice encourages computational reproducibility, since you can specify exactly which version of the code was used to produce the results in your publication. `10.5281/zenodo.4455879` is the DOI which cites _all_ versions of the code; it will always resolve to the latest release. Click on the Zenodo badge above to get access to BibTeX, etc. info related to these DOIs, e.g.:

```
@software{athena,
  author       = {Athena++ development team},
  title        = {{PrincetonUniversity/athena-public-version:
                   Athena++ v21.0}},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {21.0},
  doi          = {10.5281/zenodo.4455880},
  url          = {https://doi.org/10.5281/zenodo.4455880}
}
```

# Introduction to TIGRIS

Welcome! Thank you for considering contributing to TIGRIS and Athena++.

**The following guidelines are mostly copy-pasted from the original [`CONTRIBUTING.md`](https://github.com/changgoo/athena/blob/master/CONTRIBUTING.md) document in the Athena++ repo with small modifications for the TIGRIS project.**

The guidelines in this document are meant to help make the development of Athena++ straightforward and effective. They are a set of best practices, not strict rules, and this document may be modified at any time. Navigating the code can be daunting for new users, so if anything is unclear, please let us know!

[Doxygen Documentation](https://changgoo.github.io/athena) is currently available for public access and automatically deployed whenever push or PR is made to `tigris-master`.
<!-- ### Table of Contents -->

## Resources and quick links
* The latest development version of TIGRIS is hosted in the private fork [changgoo/athena](https://github.com/changgoo/athena) of the latest development version of Athena++ hosted in the public [PrincetonUniversity/athena](https://github.com/PrincetonUniversity/athena) GitHub repository.
* The predecessor version, TIGRESS, built on [PrincetonUniversity/Athena-Cversion](https://github.com/PrincetonUniversity/Athena-Cversion) is hosted in the private [PrincetonUniversity/Athena-TIGRESS](https://github.com/PrincetonUniversity/Athena-TIGRESS) GitHub repository.
  * [Athena 4.2 Documentation](https://princetonuniversity.github.io/Athena-Cversion/AthenaDocs) is hosted on GitHub Pages.
  * The [Athena 4.2 Test Page](https://www.astro.princeton.edu/~jstone/Athena/tests/) contains useful algorithm test results.
* The [TIGRESS website](http://changgoo.github.io/tigress-web/index.html) is hosted by [GitHub Pages](https://pages.github.com/).
* The [Athena++ website](https://princetonuniversity.github.io/athena/index.html) is hosted by [GitHub Pages](https://pages.github.com/).
* The latest version of the [Athena++ documentation](https://github.com/PrincetonUniversity/athena/wiki) is hosted as a GitHub Wiki attached to the private repository.
* The TIGRIS Slack workspace is located at [tigrepp.slack.com](https:://tigrepp.slack.com).
* The Athena++ Slack workspace is located at [athena-pp.slack.com](https:://athena-pp.slack.com).
<!-- Could add links to New PR, New Issue, Issue Labels e.g. current "bugs" -->

# How to contribute
There are many ways to contribute! We welcome feedback, [documentation](#documentation), tutorials, scripts, [bug reports](#bug-reports), [feature requests](#suggesting-enhancements), and [quality pull requests](#pull-requests).

## Using the issue tracker
Both [bug reports](#bug-reports) and [feature requests](#suggesting-enhancements) should use the [GitHub issue tracker](https://github.com/changgoo/athena/issues).

Please do not file an issue to ask a question on code usage.

### Bug reports
[Open a new Issue](https://github.com/changgoo/athena/issues/new)

Fill out the relevant sections of the [`ISSUE_TEMPLATE.md`](https://github.com/changgoo/athena/blob/master/.github/ISSUE_TEMPLATE.md) to the best of your ability when submitting a new issue.

### Suggesting enhancements
Feature requests are welcome, and are also tracked as [GitHub issues]( https://guides.github.com/features/issues/).

Please understand that we may not be able to respond to all of them because of limited resources.

## Submitting changes
Some requirements for code submissions:
- Athena++ is licensed under the BSD 3-Clause License; contributions must also use the BSD-3 license.
- The code must be commented and well documented, see [Documentation](#documentation).
- The Athena++ Wiki has a [Style Guide](https://github.com/PrincetonUniversity/athena/wiki/Style-Guide) section in the Programmer Guide. Please follow these conventions as closely as possible in order to promote consistency in the codebase.
- When implementing new functionality, add a regression test. See [Testing and continuous integration (CI)](#testing-and-continuous-integration-CI).
- If your submission fixes an issue in the [issue tracker](https://github.com/changgoo/athena/issues), please reference the issue # in the pull request title or commit message, for example:
```
Fixes #42
```

The below instructions assume a basic understanding of the Git command line interface.
If you are new to Git or a need a refresher, the [Atlassian Bitbucket Git tutorial](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud) and the [Git documentation](https://git-scm.com/) are helpful resources.

**As you (who are in the TIGRIS team) all expected to have write permission to the repository, you can create branches directly. But, you can also follow the guideline below.**

The easiest way to contribute to TIGRIS is to fork the repository to your GitHub account, create a branch on your fork, and make your changes there. When the changes are ready for submission, open a pull request (PR) on the Athena++ repository. The workflow could be summarized by the following commands:
1. Fork the repository to your GitHub account (only once) at https://github.com/changgoo/athena/fork
2. Clone a local copy of your private fork:
```
git clone https://github.com/<username>/athena ./athena-<username>
```
3. Create a descriptively-named feature branch on the fork:
```
cd athena-<username>
git checkout -b cool-new-feature
```
4. Commit often, and in logical groups of changes.
  * Use [interactive rebasing](https://help.github.com/articles/about-git-rebase/) to clean up your local commits before sharing them to GitHub.
  * Follow [commit message guidelines](https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines); see also [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/).
```
git add src/modified_file.cpp
# Use your editor to format the commit message
git commit -v
```
5. Push your changes to your remote GitHub fork:
```
git push -u origin cool-new-feature
```
6. When your branch is complete and you want to add it to TIGRIS, open a new pull request to `tigris-master`.
**Note that the default base repository and branch to which the pull requests are merged is `PrincetonUniversity/athena` and `master`. Please change the `base repository` and `branch` to `changgoo/athena` and `tigris-master`.**

### Forks and branches
The use of separate branches for both new features and bug fixes, no matter how small, is highly encouraged. Committing directly to `master` branch should be kept to a minimum. [Branches in Git are lightweight](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell), and merging small branches should be painless.

For the majority of development, users should use personal forks instead of branches on [changgoo/athena](https://github.com/changgoo/athena) (especially for larger development projects). The shared Athena++ repository should only contain a restricted set of main feature branches and temporary hotfix branches at any given time. <!-- consider reaching out to Athena++ developers before starting any significant PR/feature development to see if anyone is working on it or if we would consider merging it into Athena++-->

To update your private fork with changes from [changgoo/athena](https://github.com/changgoo/athena), from the `master` branch on a cloned copy of the forked repo:
1. Add a remote named `upstream` for the original Athena++ repository:
```
git remote add upstream https://github.com/changgoo/athena
```
2. Fetch the updates from the original Athena++ repository:
```
git fetch upstream
```
3. Merge the new commits into your forked `master`:
```
git merge --ff-only upstream/master
```
will work if you have not committed directly to your forked `master` branch.

If you have modified your forked `master` branch, the last two steps could be replaced by:
```
git pull --rebase upstream master
```
See [Developing on shared `branch`](#developing-on-shared-branch).

### Developing on shared `branch`
There are a few practices that should be followed when committing changes to a collaborative `branch` on [changgoo/athena](https://github.com/changgoo/athena) in order to avoid conflicts and headaches. These guidelines especially apply to developing on the fast changing `master` branch for those users with Admin permissions.

If you commit to an outdated local copy of `branch` (i.e. someone else has pushed changes to GitHub since you last checked), the `git push origin branch` command will be rejected by the server and prompt you to execute the `git pull` command. The default `git pull` behavior in this scenario is to create a merge-commit after you resolve any conflicts between your changes and the remote commits. However, these non-descriptive commit messages tend to clutter the repository history unnecessarily.
<!-- insert image of Network graph to compare linear and non-linear Git history -->
Most of them likely could have been avoided by either 1) doing local development on feature branches or 2) using `git pull --rebase` to perform a rebase instead of a merge when pulling conflicting updates.

If you frequently encounter such issues, it is recommended to enable the latter by default. In git versions >= 1.7.9, this can be accomplished with:
```
git config --global pull.rebase true
```

### Pull requests
When your changes are ready for submission, you may open a new pull request to `tigris-master` from a branch on the main repository (Write access) or from a branch on your private forked repository. For the latter, go to the page for your fork on GitHub, select your development branch, and click the pull request button. Fill out the relevant sections of the [`PULL_REQUEST_TEMPLATE.md`](https://github.com/changgoo/athena/blob/master/.github/PULL_REQUEST_TEMPLATE.md) to the best of your ability when submitting a new PR.

We will discuss the proposed changes and may request that you make modifications to your code before merging. To do so, simply commit to the feature branch and push your changes to GitHub, and your pull request will reflect these updates.

Before merging the PRs, you may be asked to squash and/or rebase some or all of your commits in order to preserve a clean, linear Git history. We will walk you through the interactive rebase procedure, i.e.
```
git rebase -i master
```

In general for Athena++, merging branches with `git merge —no-ff` is preferred in order to preserve the historical existence of the feature branch.

After the pull request is closed, you may optionally want to delete the feature branch on your local and remote fork via the GitHub PR webpage or the command line:
```
git branch d cool-new-feature
git push origin --delete cool-new-feature
```

### Code review policy
Currently, `master` and `tigris-master` are GitHub [protected branches](https://help.github.com/articles/about-protected-branches/), which automatically:
* Disables force pushing on `master` and `tigris-master`
* Prevents `master` and `tigris-master` from being deleted

Additionally, we have enabled ["Require pull request reviews before merging"](https://help.github.com/articles/enabling-required-reviews-for-pull-requests/) to `tigris-master`. This setting ensures that all pull requests require at least 1 code review before the branch is merged to the `tigris-master` branch and effectively prohibits pushing **any** commit directly to `tigris-master`, even from users with Write access.

Only collaborators with Admin permissions (`Athena++_admin` and `changgoo`) can bypass these restrictions. The decision to force the use of branches and pull requests for all changes, no matter how small, was made in order to:
1. Allow for isolated testing and human oversight/feedback/discussion of changes
2. Promote a [readable](https://fangpenlin.com/posts/2013/09/30/keep-a-readable-git-history/), [linear](http://www.bitsnbites.eu/a-tidy-linear-git-history/), and reversible Git history for computational reproducibility and maintainability
3. Most importantly, prevent any accidental pushes to `tigris-master`
<!-- Currently set # of required reviews to 1; other options to consider enabling in the future include: -->
<!-- "Dismiss stale PR approvals when new commits are pushed" -->
<!-- "Restrict who can push to this branch" (redundant with Require PR reviews)-->
<!-- "Require status checks to pass before merging" after separating CI build steps in new GitHub Checks API-->
<!-- "Include administrators" -->


## Testing and continuous integration (CI)
Automated testing is an essential part of any large software project. The [Regression Testing](https://github.com/PrincetonUniversity/athena/wiki/Regression-Testing) page in the Athena++ Wiki describes how to use and write new tests for the framework setup in the `tst/regression/` folder. Developers should run these tests to ensure that code changes did not break any existing functionalities.

Continuous integration is currently provided by both the Princeton Jenkins server and Travis CI service. These services automatically use the [Regression Testing](https://github.com/PrincetonUniversity/athena/wiki/Regression-Testing) framework to check code functionality and code <a href="https://en.wikipedia.org/wiki/Lint_(software)">linters</a> to ensure that conventions in the [Style Guide](https://github.com/PrincetonUniversity/athena/wiki/Style-Guide) are obeyed. The details of the infrastructure setup and instructions on how to use these services are covered in the [Continuous Integration (CI)](https://github.com/PrincetonUniversity/athena/wiki/Continuous-Integration-%28CI%29) Wiki page.

## Documentation
The development repository's [documentation](https://github.com/changgoo/athena/wiki) is a [GitHub Wiki](https://help.github.com/articles/about-github-wikis/) and is written largely in Markdown. Limited math typesetting is supported via HTML. See existing Wiki source for examples, e.g. [Editing: Coordinate Systems and Meshes](https://github.com/PrincetonUniversity/athena/wiki/Coordinate-Systems-and-Meshes/_edit).

Any significant change or new feature requires accompanying documentation before being merged to `tigris-master`. While edits can be made directly using the online interface, the Wiki is a normal Git repository which can be cloned and modified. [However](https://help.github.com/articles/adding-and-editing-wiki-pages-locally/):
> You and your collaborators can create branches when working on wikis, but only changes pushed to the `master` branch will be made live and available to your readers.
