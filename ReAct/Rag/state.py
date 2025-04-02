from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        not_useful: if zero useful information return true
        documents: list of documents
    """
    question: str
    Web_Search: bool
    documents: List[str]