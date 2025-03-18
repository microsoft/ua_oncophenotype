from typing import List, Optional, Union

from langchain_core.callbacks import (
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import RunnableConfig


def get_child_config(
    run_manager: Optional[CallbackManagerForChainRun] = None,
    extra_callbacks: Union[BaseCallbackHandler, List[BaseCallbackHandler], None] = None,
) -> RunnableConfig:
    _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
    callbacks = _run_manager.get_child()
    if not isinstance(callbacks, BaseCallbackManager):
        raise ValueError(f"Expected a CallbackManager, but got {type(_run_manager)}")
    if isinstance(extra_callbacks, BaseCallbackHandler):
        extra_callbacks = [extra_callbacks]
    for extra_callback in extra_callbacks or []:
        callbacks.add_handler(extra_callback)
    return RunnableConfig(callbacks=callbacks)
