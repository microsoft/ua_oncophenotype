from typing import List, Union

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, validator


class PydanticOutputParserWithExamples(PydanticOutputParser):
    """PydanticOutputParser whose format instructions can include custom examples."""

    examples: Union[List[BaseModel], BaseModel] = []

    @validator("examples")
    def validate_examples(cls, examples):
        if isinstance(examples, BaseModel):
            return [examples]
        return list(examples)

    def get_format_instructions(self) -> str:
        instructions = super().get_format_instructions()
        if type(self.examples) is not list:
            raise ValueError(
                "examples should be a list, should be guaranteed by validator"
            )
        if self.examples:
            instructions += "\n"
        for example in self.examples:
            instructions += f"\nexample output:\n\n```\n{example.json(indent=2)}\n```\n"
        return instructions
