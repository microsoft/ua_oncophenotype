import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from langchain.chains.base import Chain
from rwd_llm.data_loaders import DatasetBase
from rwd_llm.experiment import Experiment
from rwd_llm.experiment_config import ExperimentConfig

CUR_DIR = os.path.dirname(__file__)


def test_config_experiment():
    """This just tests that the config can be loaded and the experiment instantiated."""

    # need to set OPENAI_API_KEY and OPENAI_API_VERSION to something, otherwise it will fail
    os.environ["OPENAI_API_KEY"] = "fake"
    os.environ["OPENAI_API_VERSION"] = "fake"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "fake"

    initialize_config_dir(
        config_dir=f"{CUR_DIR}/sample_configs",
        job_name="simple_configs_test",
        version_base="1.3",
    )
    cfg = compose(
        config_name="note_experiment_config",
        overrides=[f"+config.experiment.dataset.data_root_dir={CUR_DIR}"],
    )
    config: ExperimentConfig = instantiate(cfg.config, _convert_="partial")
    assert type(config) == ExperimentConfig
    experiment: Experiment = config.experiment
    assert type(experiment) == Experiment
    assert type(experiment.dataset) == DatasetBase
    assert isinstance(experiment.chain, Chain)
    GlobalHydra.instance().clear()


if __name__ == "__main__":
    test_config_experiment()
