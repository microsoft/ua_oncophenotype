import re
from typing import Dict, List, Tuple

from langchain.prompts import PromptTemplate


def find_format_variables(s: str, jinja: bool = False) -> List[str]:
    if jinja:
        # This pattern captures content between {{ and }}
        pattern = re.compile(r"{{([^{}]+)}}")
    else:
        # This regex captures content between { and }, but avoids {{ or }}
        pattern = re.compile(r"(?<!{){([^{}]+)}(?!})")
    return pattern.findall(s)


def get_prompt_from_message(
    message: str, possible_partial_variables: Dict[str, str], jinja: bool = False
) -> Tuple[PromptTemplate, List[str]]:
    """Get a prompt from a message, using possible_partial_variables to inject
    partial variables into the prompt. Returns the prompt and a list of partial
    variables that were used."""
    format_variables = find_format_variables(message, jinja=jinja)
    partial_variables = {}
    for variable in possible_partial_variables:
        if variable in format_variables:
            format_variables.remove(variable)
            partial_variables[variable] = possible_partial_variables[variable]
    prompt = PromptTemplate(
        template=message,
        input_variables=format_variables,
        partial_variables=partial_variables,
        template_format="jinja2" if jinja else "f-string",
    )
    return prompt, list(partial_variables)


def validate_prompt_inputs(
    input_variables: List[str], found_format_variables: List[str]
):
    err = ""
    vars_not_found_in_prompt = list(set(input_variables) - set(found_format_variables))
    if vars_not_found_in_prompt:
        err += (
            "Variables specified in input_variables but not found in prompt:"
            f" {vars_not_found_in_prompt}"
        )
    vars_not_found_in_input = list(set(found_format_variables) - set(input_variables))
    if vars_not_found_in_input:
        err += (
            "Variables found in prompt but not specified in input_variables:"
            f" {vars_not_found_in_input}"
        )
    if err:
        raise ValueError(err)
