# SignatureQLearning
 Code for master thesis on a new variant of Q-learning for history-dependet domains and its application to market making.

## Installation

1. Download this repositories source code, either directly from GitHub or with git:

    ```bash
    git clone --recursive https://github.com/phaelicks/SignatureQLearning.git
    ```
    **Note** This repositiory contains https://github.com/jpmorganchase/abides-jpmc-public
    as submodule. `git clone --recursive` ensures that the download contains submodules.
    In case the directory `abides` is empty after cloning this repository, navigate to this files directory and run `git submodule update --init --recursive`.

2. Run the install script to install this repositories requirements and dependencies, including the ABIDES packages and their dependencies:
    ```
    install.sh
    ```
