import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

logger = logging.getLogger(__name__)


# class LLMCallback(LLMManagerMixin, CallbackManagerMixin):
class LLMCallMemorizer(BaseCallbackHandler):
    def __init__(self):
        self._prompt = None
        self._response = None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info("LLMCallMemorizer.on_llm_start called")
        if self._prompt is not None:
            raise ValueError("Expected exactly one call to on_llm_start")
        if len(prompts) != 1:
            raise ValueError(f"Expected exactly one prompt (found {len(prompts)})")
        self._prompt = prompts[0]

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info("LLMCallMemorizer.on_chat_model_start called")
        if self._prompt is not None:
            raise ValueError("Expected exactly one call to on_chat_model_start")
        if len(messages) != 1:
            raise ValueError(f"Expected exactly one prompt (found {len(messages)})")
        self._prompt = messages[0]

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info("LLMCallMemorizer.on_llm_end called")
        if self._response is not None:
            raise ValueError("Expected exactly one call to on_llm_end")
        if len(response.generations) != 1:
            raise ValueError(
                f"Expected exactly one response (found {len(response.generations)})"
            )
        self._response = response.generations[0][0].text


DEFAULT_INSPECTION_PROMPT = """
The following question was answered by a large language model:


PROMPT:
{prompt}
ANSWER:
{response}

Review the evidence and reasoning carefully. Is the ANSWER correct?
"""


class SelfInspectionChain(Chain):
    """Chain that wraps another chain, and checks the output of the wrapped chain"""

    chain_to_inspect: LLMChain
    inspection_chain: LLMChain
    inspection_chain_prompt_key: str = "prompt"
    inspection_chain_response_key: str = "response"

    @classmethod
    def from_chain(
        cls,
        chain_to_inspect: LLMChain,
        # inspection_chain: LLMChain,
        inspection_prompt: str = DEFAULT_INSPECTION_PROMPT,
        inspection_prompt_llm: Optional[LLMChain] = None,
        inspection_chain_prompt_key: str = "prompt",
        inspection_chain_response_key: str = "response",
        inspection_chain_output_key: str = "inspection_output",
    ) -> "SelfInspectionChain":
        if inspection_prompt_llm is None:
            inspection_prompt_llm = chain_to_inspect.llm
        inspection_chain = LLMChain(
            prompt=PromptTemplate(
                template=inspection_prompt,
                input_variables=[
                    inspection_chain_prompt_key,
                    inspection_chain_response_key,
                ],
            ),
            llm=inspection_prompt_llm,
            output_key=inspection_chain_output_key,
        )
        return cls(
            chain_to_inspect=chain_to_inspect,
            inspection_chain=inspection_chain,
            inspection_chain_prompt_key=inspection_chain_prompt_key,
            inspection_chain_response_key=inspection_chain_response_key,
        )

    @property
    def input_keys(self) -> List[str]:
        """Either user-specified, or whatever keys the root prompt expects."""
        return self.chain_to_inspect.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Expect output keys"""
        return self.chain_to_inspect.output_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        mem_tracer = LLMCallMemorizer()
        callbacks = _run_manager.get_child()
        if not isinstance(callbacks, BaseCallbackManager):
            raise ValueError(
                f"Expected a CallbackManager, but got {type(_run_manager)}"
            )
        # add the new tracer
        callbacks.add_handler(mem_tracer)
        outputs = self.chain_to_inspect(
            inputs, callbacks=callbacks, return_only_outputs=True
        )

        inspection_inputs = {
            self.inspection_chain_prompt_key: mem_tracer._prompt,
            self.inspection_chain_response_key: mem_tracer._response,
        }

        callbacks = _run_manager.get_child()
        inspection_output = self.inspection_chain(
            inspection_inputs, callbacks=callbacks, return_only_outputs=True
        )
        inspection_output.update(outputs)
        print(inspection_output)
        return inspection_output
