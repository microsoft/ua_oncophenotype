import logging
import os
from typing import Union

import hydra
import hydra.core
from omegaconf import DictConfig, ListConfig
from rwd_llm.experiment_config import parse_config

from .utils import get_by_path

logger = logging.getLogger(__name__)


INDENT_SIZE = 2
INDENT = " " * INDENT_SIZE


def print_cfg(cfg: Union[DictConfig, ListConfig], indent: int = 0):
    """This prints 'interpolated' (${...}) values, which is useful for debugging."""
    if isinstance(cfg, ListConfig):
        for value in cfg:
            if isinstance(value, (DictConfig, ListConfig)):
                print(INDENT * indent + "-")
                print_cfg(value, indent + 1)
            else:
                print(INDENT * indent + "- " + str(value))
    elif isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, (DictConfig, ListConfig)):
                print(INDENT * indent + str(key) + ": ")
                print_cfg(value, indent + 1)
            else:
                print(INDENT * indent + str(key) + ": " + str(value))
    else:
        logger.warning(f"Got non-Config value {type(cfg)} in print_cfg")
        print(INDENT * indent + str(cfg))


# must specify a config_dir via the command line
@hydra.main(config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    # tell hydra to raise exceptions instead of catching and reporting
    os.environ["HYDRA_FULL_ERROR"] = "1"

    logging.basicConfig(level=logging.DEBUG)

    if "print" in cfg:
        # standard values meaning 'just print the config'
        if cfg.print not in ["", None, "true", "True", "1", 1, True]:
            # try to get sub-path in config based on value of print
            # e.g. +print=foo.bar will print cfg.foo.bar
            try:
                sub_cfg = get_by_path(cfg, cfg.print)
            except Exception as e:
                print(
                    f"Failed to get config path {cfg.print}, was this meant to be a"
                    " key? If not use '+print=true'"
                )
                raise e
            cfg = sub_cfg
        print_cfg(cfg)
        return

    # config can either bet at the top level or nested under "config"
    config = parse_config(cfg)
    config.run()


if __name__ == "__main__":
    my_app()
