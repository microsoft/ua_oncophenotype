from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from rwd_llm.tests.utils import FakeLLM

from ..self_inspection_chain import SelfInspectionChain


def test_normal_usage():
    question = "What is the largest animal ever to have lived on Earth?"
    FakeLLM.set_answers(["the bumblebee", "No, the largest animal is the blue whale."])

    prompt = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template(question)]
    )
    inner_chain = LLMChain(prompt=prompt, llm=FakeLLM())
    wrapper_chain = SelfInspectionChain.from_chain(chain_to_inspect=inner_chain)
    result = wrapper_chain({})
    print(result)


if __name__ == "__main__":
    test_normal_usage()
