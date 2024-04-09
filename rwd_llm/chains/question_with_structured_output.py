import logging
from typing import List, Optional, Union

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import AIMessage
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from rwd_llm.utils import get_prompt_from_message
from rwd_llm.utils.prompt_utils import validate_prompt_inputs

logger = logging.getLogger(__name__)


def chat_prompt_with_structured_output(
    question: str,
    parser: PydanticOutputParser,
    instructions: Optional[str] = None,
    preamble: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    format_instructions_variable: str = "format_instructions",
    examples: Optional[Union[List[dict], BaseExampleSelector]] = None,
    example_input_vars: Optional[List[str]] = None,
    example_output_var: str = "result",
    example_indent: int = 2,
    jinja: bool = False,
) -> ChatPromptTemplate:
    """
    Create a chat prompt with structured output and optional few-shot examples.

    The format is: System Message: {preamble} System Message: {instructions} for each
    few-shot example:
        Human Message: {question} AI Message: {result}
    Human Message: {question}

    Args:
        question: The question to ask the user. parser: The output parser to use.
        instructions: Optional instructions to give the user. preamble: Optional
        preamble to give the user.
        input_variables: Optional list of input variables to validate against.  If not
          given, the input variables will be determined from the prompts.
        format_instructions_variable: The variable name to use for the format
          instructions.
        examples: Optional list of few-shot examples.
        example_input_vars: The variables to use for the example input.
        example_output_var: The variable name to use for the example output.
        example_indent: The number of spaces to use for indentation in the example
          output.
    """
    messages = []
    found_format_variables = []
    examples = examples or []

    # the JSON schema has lots of curly braces ({}), so we need to inject them as
    # partial variables
    partial_variables = {format_instructions_variable: parser.get_format_instructions()}
    if preamble is not None:
        template, _ = get_prompt_from_message(preamble, partial_variables, jinja=jinja)
        found_format_variables.extend(template.input_variables)
        msg = SystemMessagePromptTemplate(prompt=template)
        messages.append(msg)

    if instructions is not None:
        template, _ = get_prompt_from_message(
            instructions, partial_variables, jinja=jinja
        )
        found_format_variables.extend(template.input_variables)
        msg = SystemMessagePromptTemplate(prompt=template)
        messages.append(msg)

    question_template, _ = get_prompt_from_message(
        question, partial_variables, jinja=jinja
    )
    # add few shot examples
    if isinstance(examples, BaseExampleSelector):
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            # Which variable(s) will be passed to the example selector.
            input_variables=example_input_vars or [],
            example_selector=examples,
            # Define how each example will be formatted.
            # In this case, each example will become 2 messages:
            # 1 human, and 1 AI
            example_prompt=(
                HumanMessagePromptTemplate(prompt=question_template)
                + AIMessagePromptTemplate.from_template(f"{{{example_output_var}}}")
            ),
        )
        messages.append(few_shot_prompt)
    else:
        for example in examples:
            result = example.pop(example_output_var)
            if example_input_vars and set(example_input_vars) != set(example.keys()):
                logger.warning(
                    "Few-shot example does not contain the expected input variables."
                )
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
