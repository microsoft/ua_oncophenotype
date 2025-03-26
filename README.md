
# Quick Start

1. Create the environment
    ```sh
    conda create -f ua_oncophenotype_env
    conda activate ua_oncophenotype_env
    # the environment contains all of the dependencies, but we'll install the package in 
    # editable mode
    pip install -e ua_oncophenotype
    # install the test requirements
    pip install -r test_requirements.txt
    ```
1. Run the unit tests
    ```sh
    ./run_checks.py -t
    ```
1. Run the sample scripts
    1. Copy the sample config and edit to point to your AOAI instance
        ```sh
        cd ua_oncophenotype/tests/sample_configs/config/openai_config
        cp sample_openai_config.yaml local_openai_config.yaml
        ```
    1. Edit the local_openai_config.yaml to point to your AOAI resource and add the
       appropriate key
       Note: Alternatively copy and modify `sample_openai_identity_auth_config.yaml`
    1. Run the sample script
        ```sh
        # go to ua_oncophenotype/tests
        cd ../../..
        ./run_test_experiment.sh
        ```

