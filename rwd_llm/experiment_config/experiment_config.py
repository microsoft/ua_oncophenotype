import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from haikunator import Haikunator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..experiment import Experiment
from ..llms.llm_utils import OpenAIConfig, setup_openai_from_dotenv
from ..utils import get_config_dir, get_overrides
from .component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


def haikunate(token_length: int = 4, delimiter: str = "-") -> str:
    return Haikunator().haikunate(token_length=token_length, delimiter=delimiter)


OmegaConf.register_new_resolver("haikunate", haikunate, use_cache=True)


def _instantiate_hook(hook_config) -> Callable[[], None]:
    try:
        hook_class = hook_config["_target_"]
    except KeyError:
        raise KeyError(f"Missing _target_ key in hook config {hook_config}")
    logger.info(f"Instantiating experiment hook of type {hook_class}")
    return instantiate(hook_config, _convert_="partial")


@dataclass
class ExperimentConfig:
    experiment: Experiment
    output_dir: str
    resources: list = field(default_factory=list)
    openai_config: Optional[OpenAIConfig] = None
    pre_run_hooks: List[Dict] = field(default_factory=list)
    post_run_hooks: List[Dict] = field(default_factory=list)

    def parse_config(self):
        """The config is initially only "shallow" parsed, so elements are still
        OmegaConf DictConfig/ListConfig format.  This function should finish
        instantiating the config object.  This is necessary to enforce instantiation
        ordering regardless of how the config is written."""

        # setup openai either from config or from env/dotenv, needs to be done before
        # instantiating OpenAI LLM to avoid errors
        if self.openai_config:
            openai_config: OpenAIConfig = instantiate(self.openai_config)
            openai_config.setup_openai()
            setup_openai_from_dotenv(raise_error=False)
        else:
            setup_openai_from_dotenv(raise_error=True)

        # load any resources prior to instantiating the experiment
        if self.resources is None:
            self.resources = []
        for resource in self.resources:
            if len(resource) != 1:
                raise ValueError(
                    f"Invalid resource {resource}, expected single-entry dict"
                )
            name, resource = list(resource.items())[0]
            logger.info(f"Instantiating resource {name}")
            ComponentRegistry.register(name, instantiate(resource, _convert_="partial"))

        logger.info("Instantiating pre-run hooks")
        self.pre_run_hooks = [_instantiate_hook(hook) for hook in self.pre_run_hooks]
        logger.info("Instantiating post-run hooks")
        self.post_run_hooks = [_instantiate_hook(hook) for hook in self.post_run_hooks]

        for pre_run_hook in self.pre_run_hooks:
            logger.info(f"Running hook of type {type(pre_run_hook)}")
            pre_run_hook()

        # _convert_="partial" required so that pydantic classes don't throw exception
        # from OmegaConf ListConfig/DictConfig objects
        logger.info("Instantiating the experiment")
        self.experiment = instantiate(self.experiment, _convert_="partial")

    def copy_inputs(self):
        """Copy original config directory and overrides to output directory."""
        config_dir = get_config_dir()
        overrides = get_overrides()
        configs_out_dir = os.path.join(self.output_dir, "configs")
        shutil.copytree(config_dir, configs_out_dir)
        with open(os.path.join(self.output_dir, "overrides.txt"), "w") as f:
            f.write(overrides)

    def run(self, allow_missing_hydra_configs: bool = False):
        """This is called by run_experiment.py. Subclasses should override _run()."""
        logger.info("Copying inputs to run directory")
        try:
            self.copy_inputs()
        except ValueError as err:
            if allow_missing_hydra_configs:
                logger.warning(f"Error copying inputs: {err}")
            else:
                raise err
        logger.info("running experiment config")
        self._run()
        logger.info("Running post-run hooks")
        for post_run_hook in self.post_run_hooks:
            logger.info(f"Running hook of type {type(post_run_hook)}")
            post_run_hook()

    def _run(self):
        """Subclasses can override this method for custom functionality"""
        logger.info("running experiment")
        results = self.experiment.run()
        logger.info("saving experiment results")
        results.write(self.output_dir)


def parse_config(cfg: DictConfig) -> ExperimentConfig:
    if "config" in cfg:
        cfg = cfg.config

    config = instantiate(cfg, _recursive_=False)
    config.parse_config()
    return config
