#!/bin/bash

set -x
set -e

# to print the full config (inclding expanded OmegaConf interpolations), add
# "+print=true" to the command line

### Uncomment these lines to run the experiment with the langchain handler
# export LANGCHAIN_HANDLER=langchain
# export LANGCHAIN_TRACING=true
### Make sure this session actually exists. You can create a new session in the UI.
# export LANGCHAIN_SESSION=default
# export LANGCHAIN_SESSION=test01

# OPENAI_CONFIG=sample_openai_config
OPENAI_CONFIG=local_openai_config
# OPENAI_CONFIG=local_openai_alt_config

# for creating the index add the argument: config/experiment/chain/index=build_index 
# EXPERIMENT=patient_index_retrieval_experiment_config.yaml

EXPERIMENT=patient_retrieval_experiment_config.yaml
# EXPERIMENT=note_experiment_config.yaml
# EXPERIMENT=patient_tumor_site_retrieval_config.yaml


# get the location of this script, which is the data_root_dir for the sample data
TESTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$TESTS_DIR
# DATA_DIR=$DATA_ROOT_DIR

python -m rwd_llm.run_experiment \
    -cp tests/sample_configs \
    -cn $EXPERIMENT \
    +dataset.data_root_dir=$DATA_DIR \
    config/openai_config=$OPENAI_CONFIG \
    +config.experiment.data_runner.raise_exceptions=true \
    $*
