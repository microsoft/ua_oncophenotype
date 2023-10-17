import os
from dataclasses import dataclass
from typing import Optional

import openai
from dotenv import find_dotenv, load_dotenv


def _get_var(var: str, raise_error: bool = True) -> Optional[str]:
    val = os.environ.get(var, None)
    if val is None and raise_error:
        raise ValueError(f"Required environment variable {var} is not set.")
    return val


@dataclass
class OpenAIConfig:
    api_base: str
    api_version: str
    api_key: str
    api_type: Optional[str] = None

    def setup_openai(self):
        if self.api_type:
            openai.api_type = self.api_type
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        _set_openai_env_vars()


def _set_openai_env_vars():
    if openai.api_base:
        os.environ["OPENAI_API_BASE"] = openai.api_base
    if openai.api_version:
        os.environ["OPENAI_API_VERSION"] = openai.api_version
    if openai.api_key:
        os.environ["OPENAI_API_KEY"] = openai.api_key
    if openai.api_type:
        os.environ["OPENAI_API_TYPE"] = openai.api_type


def setup_openai_from_dotenv(raise_error: bool = True):
    # test the current path for a .env file
    dotenv_path = find_dotenv(usecwd=True) or None
    # if dotenv wasn't found (dotenv_path is None), this will look relative to the
    # python script
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    api_type = _get_var("OPENAI_API_TYPE", raise_error=raise_error)
    if api_type:
        openai.api_type = api_type
    api_key = _get_var("OPENAI_API_KEY", raise_error=raise_error)
    if api_key:
        openai.api_key = api_key
    api_base = _get_var("OPENAI_API_BASE", raise_error=raise_error)
    if api_base:
        openai.api_base = api_base
    api_version = _get_var("OPENAI_API_VERSION", raise_error=raise_error)
    if api_version:
        openai.api_version = api_version
    _set_openai_env_vars()
