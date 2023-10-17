import os
import pprint
from typing import Dict

import hydra.core.hydra_config
from omegaconf import OmegaConf

pp = pprint.PrettyPrinter(indent=4)


def print_hydra_config():
    """Print the hydra config (not the user config) to stdout."""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    print(pp.pprint(OmegaConf.to_container(hydra_config)))


def get_out_dir() -> str:
    """get the hydra output directory"""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    return hydra_config.run.dir


def get_hydra_out_dir() -> str:
    """get the hydra output sub-directory (${out_dir}.hydra by default)"""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    return os.path.join(get_out_dir(), hydra_config.output_subdir)


def get_config_dir() -> str:
    """Get location of config dir. This isn't foolproof, we're expecting a simple run
    where 'cd' is specified on the command line, and there aren't additional directories
    on the search path."""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    config_dirs = [
        cfg.path
        for cfg in hydra_config.runtime.config_sources
        if cfg.provider == "main"
    ]
    if len(config_dirs) != 1:
        raise ValueError(f"Expected single config dir, got {config_dirs}")
    return config_dirs[0]


def get_overrides() -> str:
    """Get a text representation of the overrides used to run this experiment. This
    relies on some internals of hydra that may break in future versions."""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    return hydra_config.job.override_dirname


def get_overrides_dict() -> Dict[str, str]:
    """Get a text representation of the overrides used to run this experiment. This
    relies on some internals of hydra that may break in future versions."""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    override_list = []
    for override in hydra_config.overrides.task:
        override_tuple = override.split("=")
        if len(override_tuple) != 2:
            raise ValueError(f"Invalid override {override}")
        override_list.append(tuple(override_tuple))
    return dict(override_list)
