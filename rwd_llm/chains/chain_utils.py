import json
from typing import Any, Dict, List, Optional

from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# define some useful constants
# Unsure is a response from the LLM
UNSURE_LABEL = "Unsure"
# Error is used to indicate a processing error (exception)
ERROR_LABEL = "ERROR"


def load_from_json(json_file: str) -> Dict[str, Any]:
    with open(json_file, "r") as f:
        return json.load(f)


def get_example_prompt_template(
    instructions: str,
    input_variables: List[str],
    chat_prompt: bool = True,
    answer_format: str = "{answer}",
    answer_key: str = "answer",
    completion_template: str = "{instructions}\n{answer_format}",
) -> BasePromptTemplate:
    if chat_prompt:
        messages: List[BaseMessagePromptTemplate] = []
        messages.append(HumanMessagePromptTemplate.from_template(instructions))
        messages.append(AIMessagePromptTemplate.from_template(answer_format))
        prompt = ChatPromptTemplate.from_messages(messages)
    else:
        template = completion_template.format(
            instructions=instructions, answer_format=answer_format
        )
        prompt = PromptTemplate(template=template, input_variables=input_variables)
    return prompt


def get_prompt_template(
    instructions: str,
    input_variables: List[str],
    preamble: Optional[str] = None,
    chat_prompt: bool = True,
    partial_variables: Optional[Dict[str, Any]] = None,
) -> BasePromptTemplate:
    if chat_prompt:
        messages: List[BaseMessagePromptTemplate] = []
        if preamble:
            messages.append(SystemMessagePromptTemplate.from_template(preamble))
        messages.append(HumanMessagePromptTemplate.from_template(instructions))
        prompt = ChatPromptTemplate.from_messages(messages)
        if partial_variables:
            for k, v in partial_variables.items():
                prompt.partial_variables[k] = v

    else:
        template = f"{preamble}\n\n{instructions}"
        prompt = PromptTemplate(template=template, input_variables=input_variables)
    return prompt
