from typing import List, Union

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, field_validator


class PydanticOutputParserWithExamples(PydanticOutputParser):
    """PydanticOutputParser whose format instructions can include custom examples."""

    examples: Union[List[BaseModel], BaseModel] = []

    @field_validator("examples")
    @classmethod
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
            instructions += (
                f"\nexample output:\n\n```\n{example.model_dump_json(indent=2)}\n```\n"
            )
        return instructions
