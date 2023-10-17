from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import AIMessage
from llm_lib.utils import get_prompt_from_message
from llm_lib.utils.prompt_utils import validate_prompt_inputs


def chat_prompt_with_structured_output(
    question: str,
    parser: PydanticOutputParser,
    instructions: Optional[str] = None,
    preamble: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    format_instructions_variable: str = "format_instructions",
    examples: Optional[List[dict]] = None,
    example_output_var: str = "result",
    example_indent: int = 2,
) -> ChatPromptTemplate:
    messages = []
    found_format_variables = []
    examples = examples or []

    # the JSON schema has lots of curly braces ({}), so we need to inject them as
    # partial variables
    partial_variables = {format_instructions_variable: parser.get_format_instructions()}
    if preamble is not None:
        template, _ = get_prompt_from_message(preamble, partial_variables)
        found_format_variables.extend(template.input_variables)
        msg = SystemMessagePromptTemplate(prompt=template)
        messages.append(msg)

    if instructions is not None:
        template, _ = get_prompt_from_message(instructions, partial_variables)
        found_format_variables.extend(template.input_variables)
        msg = SystemMessagePromptTemplate(prompt=template)
        messages.append(msg)

    question_template, _ = get_prompt_from_message(question, partial_variables)
    # add few shot examples
    for example in examples:
        result = example.pop(example_output_var)
        result_obj = parser.pydantic_object.parse_obj(result)
        msg = HumanMessagePromptTemplate(prompt=question_template).format(**example)
        messages.append(msg)
        msg = AIMessage(content=result_obj.json(indent=example_indent))
        messages.append(msg)
    # add final question
    found_format_variables.extend(question_template.input_variables)
    msg = HumanMessagePromptTemplate(prompt=question_template)
    messages.append(msg)

    # validate inputs
    if not input_variables:
        input_variables = found_format_variables
    else:
        validate_prompt_inputs(input_variables, found_format_variables)

    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt
