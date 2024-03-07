from langchain.docstore.document import Document

from ..document_filter_chain import DocumentFilterChain


def test_document_filter_chain():
    # Create a sample document
    doc1 = Document(page_content="sample content", metadata={"my_key": "the_val"})
    doc2 = Document(page_content="sample content", metadata={"my_key": "other_val"})

    # Set up the input
    inputs = {"docs": [doc1, doc2]}

    # test allow_list
    allow_list = {"my_key": ["hello", "the_val"]}
    chain = DocumentFilterChain(allow_list=allow_list)
    expected_output = {"docs": [doc1]}
    output = chain(inputs)
    assert output == expected_output

    # test deny_list
    deny_list = {"my_key": ["the_val", "goodbye"]}
    chain = DocumentFilterChain(deny_list=deny_list)
    expected_output = {"docs": [doc2]}
    output = chain(inputs)
    assert output == expected_output


if __name__ == "__main__":
    test_document_filter_chain()
