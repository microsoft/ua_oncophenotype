import os
from dataclasses import dataclass

from azure.identity import DefaultAzureCredential

from .llm_utils import OpenAIConfigBase


@dataclass
class OpenAIAADConfig(OpenAIConfigBase):
    api_base: str
    api_version: str
    api_type: str = "azure_ad"

    def setup_openai(self):
        if self.api_base:
            os.environ["AZURE_OPENAI_ENDPOINT"] = self.api_base
        if self.api_version:
            os.environ["OPENAI_API_VERSION"] = self.api_version
        os.environ["OPENAI_API_TYPE"] = self.api_type
        credential = DefaultAzureCredential()
        token = credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token
        os.environ["OPENAI_API_KEY"] = token
