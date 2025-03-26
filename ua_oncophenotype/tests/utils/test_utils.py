from typing import ClassVar, List, Optional

from langchain.chains.base import Chain
from langchain_core.language_models import LLM
from pydantic import BaseModel


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes."""

    temperature: float = 0.0
    n: int = 1
    deployment_name: str = "fake"
    best_of: Optional[int] = None
    call_idx: ClassVar[int] = 0
    _answers: ClassVar[List[str]] = []

    @classmethod
    def set_answers(cls, answers: List[str]):
        cls.call_idx = 0
        cls._answers = answers

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM."""
        print(f"FakeLLM called with prompt: {prompt}, call_idx: {self.call_idx}")
        the_answer = self._answers[self.call_idx]
        FakeLLM.call_idx += 1
        return the_answer
