# 👐 Contributing to Transformers Interpret

First off, thank you for even considering contributing to this package, every contribution big or small is greatly appreciated. Community contributions are what keep projects like this fueled and constantly improving, so a big thanks to you!

Below are some sections detailing the guidelines we'd like you to follow to make your contribution as seamless as possible.

- [Code of Conduct](#coc)
- [Asking a Question and Discussions](#question)
- [Issues, Bugs, and Feature Requests](#issue)
- [Submission Guidelines](#submit)
- [Code Style and Formatting](#code)

## 📜 <a name="coc"></a> Code of Conduct

As contributors and maintainers of Transformers Interpret, we pledge to respect everyone who contributes by posting issues, updating documentation, submitting pull requests, providing feedback in comments, and any other activities.

Communication within our community must be constructive and never resort to personal attacks, trolling, public or private harassment, insults, or other unprofessional conduct.

We promise to extend courtesy and respect to everyone involved in this project regardless of gender, gender identity, sexual orientation, disability, age, race, ethnicity, religion, or level of experience. We expect anyone contributing to the project to do the same.

If any member of the community violates this code of conduct, the maintainers of Transformers Interpret may take action, removing issues, comments, and PRs or blocking accounts as deemed appropriate.

If you are subject to or witness unacceptable behavior, or have any other concerns, please drop us a line at [charlespierse@gmail.com](mailto://charlespierse@gmail.com)

## 🗣️ <a name="question"></a> Got a Question ? Want to start a Discussion ?

We would like to use [Github discussions](https://github.com/cdpierse/transformers-interpret/discussions) as the central hub for all community discussions, questions, and everything else in between. While Github discussions is a new service (as of 2021) we believe that it really helps keep this repo as one single source to find all relevant information. Our hope is that this page functions as a record of all the conversations that help contribute to the project's development.

We also highly encourage general and theoretical discussion. Much of the Transformers Interpret package implements algorithms in the field of explainable AI (XAI) which is itself very nascent, so discussions around the topic are very welcome. Everyone has something to learn and to teach !

If you are new to [Github discussions](https://github.com/cdpierse/transformers-interpret/discussions) it is a very similar experience to Stack Overflow with an added element of general discussion and discourse rather than solely being question and answer based.

## 🪲 <a name="issue"></a> Found an Issue or Bug?

If you find a bug in the source code, you can help us by submitting an issue to our [Issues Page](https://github.com/cdpierse/transformers-interpret/issues). Even better you can submit a Pull Request if you have a fix.

See [below](#submit) for some guidelines.

##  ✉️  <a name="submit"></a> Submission Guidelines

### Submitting an Issue

Before you submit your issue search the archive, maybe your question was already answered. If you feel like your issue is not specific and more of a general question about a design decision, or algorithm implementation maybe start a [discussion](https://github.com/cdpierse/transformers-interpret/discussions) instead, this helps keep the issues less cluttered and encourages more open ended conversation.

If your issue appears to be a bug, and hasn't been reported, open a new issue.
Help us to maximize the effort we can spend fixing issues and adding new
features, by not reporting duplicate issues. Providing the following information will increase the
chances of your issue being dealt with quickly:

- **Describe the bug** - A clear and concise description of what the bug is.
- **To Reproduce**- Steps to reproduce the behavior.
- **Expected behavior** - A clear and concise description of what you expected to happen.
- **Environment**
  - Transformers Interpret version
  - Python version
  - OS
- **Suggest a Fix** - if you can't fix the bug yourself, perhaps you can point to what might be
  causing the problem (line of code or commit)

When you submit a PR you will be presented with a PR template, please fill this in as best you can.

### Submitting a Pull Request

Before you submit your pull request consider the following guidelines:

- Search [GitHub](https://github.com/cdpierse/transformers-interpret/pulls) for an open or closed Pull Request
  that relates to your submission. You don't want to duplicate effort.
- Make your changes in a new git branch, based off master branch:

  ```shell
  git checkout -b my-fix-branch master
  ```

- There are three typical branch name conventions we try to stick to, there are of course always exceptions to the rule but generally the branch name format is one of:

  - **fix/my-branch-name** - this is for fixes or patches
  - **docs/my-branch-name** - this is for a contribution to some form of documentation i.e. Readme etc
  - **feature/my-branch-name** - this is for new features or additions to the code base

- Create your patch, **including appropriate test cases**.
  - **A note on tests**: This package is in a odd position in that is has some heavy external dependencies on the Transformers package, Pytorch, and Captum. In order to adequately test all the features of an explainer it must often be tested alongside a model, we do our best to cover most models and edge cases however this isn't always possible given that some model's have their own quirks in implementations that break convention. We don't expect 100% test coverage but generally we'd like to keep it > 90%.
- Follow our [Coding Rules](#rules).
- Avoid checking in files that shouldn't be tracked (e.g `dist`, `build`, `.tmp`, `.idea`). We recommend using a [global](#global-gitignore) gitignore for this.
- Before you commit please run the test suite and coverage and make sure all tests are passing.

  ```shell
  coverage run -m pytest -s -v && coverage report -m
  ```
- Format your code appropriately:
  * This package uses [black](https://black.readthedocs.io/en/stable/) as its formatter. In order to format your code with black run ```black . ``` from the root of the package.  

- Commit your changes using a descriptive commit message.

  ```shell
  git commit -a
  ```

  Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

* Push your branch to GitHub:

  ```shell
  git push origin fix/my-fix-branch
  ```

* In GitHub, send a pull request to `transformers-interpret:dev`.
* If we suggest changes then:

  - Make the required updates.
  - Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ```shell
    git rebase master -i
    git push origin fix/my-fix-branch -f
    ```

That's it! Thank you for your contribution!


### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the main (upstream) repository:

- Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

  ```shell
  git push origin --delete my-fix-branch
  ```

- Check out the master branch:

  ```shell
  git checkout master -f
  ```

- Delete the local branch:

  ```shell
  git branch -D my-fix-branch
  ```

- Update your master with the latest upstream version:

  ```shell
  git pull --ff upstream master
  ```

## ✅ <a name="rules"></a> Coding Rules

We generally follow the [Google Python style guide](http://google.github.io/styleguide/pyguide.html).

----

*This guide was inspired by the [Firebase Web Quickstarts contribution guidelines](https://github.com/firebase/quickstart-js/blob/master/CONTRIBUTING.md) and [Databay](https://github.com/Voyz/databay/edit/master/CONTRIBUTING.md)*