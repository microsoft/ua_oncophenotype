from pydantic import BaseModel
from ua_oncophenotype.example_selectors import InputExampleSelector
from ua_oncophenotype.output_parsers import PydanticOutputParserWithExamples

from ..question_with_structured_output import chat_prompt_with_structured_output


class MyStructuredOutput(BaseModel):
    reasoning: str
    confidence: int
    answer: str


example = MyStructuredOutput(
    reasoning="The reasoning leading to the answer", confidence=5, answer="The answer"
)

parser = PydanticOutputParserWithExamples(
    pydantic_object=MyStructuredOutput, examples=example
)


def get_examples():
    return [
        {
            "question": "What is the answer to life, the universe, and everything?",
            "response": MyStructuredOutput(
                reasoning="Because Douglas Adams said so.", confidence=5, answer="42"
            ).model_dump(),
        },
    ]


def test_chat_prompt_with_structured_output():
    preamble = "You are a helpful question answering AI."
    instructions = """
    Answer the user's question with a structured output.  In addition to the answer,
    include reasoning and your confidence in the answer on a scale from 1 (not
    confident) to 5 (very confident). Your answer should be in the form of a JSON object
    with the following structure:

    {the_formatting_instructions}
    """
    question = "{question}"
    prompt = chat_prompt_with_structured_output(
        preamble=preamble,
        instructions=instructions,
        question=question,
        parser=parser,
        format_instructions_variable="the_formatting_instructions",
        example_output_var="response",
        examples=get_examples(),
    )
    p = prompt.format_messages(
        **{"question": 'Who wrote "The Hitchhiker\'s Guide to the Galaxy"?'}
    )
    assert len(p) == 5


def test_chat_prompt_with_structured_output_dynamic_examples():
    preamble = "You are a helpful question answering AI."
    instructions = """
    Answer the user's question with a structured output.  In addition to the answer,
    include reasoning and your confidence in the answer on a scale from 1 (not
    confident) to 5 (very confident). Your answer should be in the form of a JSON object
    with the following structure:

    {the_formatting_instructions}
    """
    question = "{question}"
    prompt = chat_prompt_with_structured_output(
        preamble=preamble,
        instructions=instructions,
        question=question,
        parser=parser,
        format_instructions_variable="the_formatting_instructions",
        example_input_vars=["examples"],
        example_output_var="response",
        examples=InputExampleSelector(examples_input_key="examples"),
    )
    p = prompt.format_messages(
        **{
            "examples": get_examples(),
            "question": 'Who wrote "The Hitchhiker\'s Guide to the Galaxy"?',
        }
    )
    assert len(p) == 5


if __name__ == "__main__":
    test_chat_prompt_with_structured_output()
    test_chat_prompt_with_structured_output_dynamic_examples()
