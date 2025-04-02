from typing import Any, Dict
from state import GraphState
from grader import retrieval_grader

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["not_useful"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
        )
        return "No required information"
    else:
        print("---DECISION: Use retrieve---")
        return state["documents"]


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    #Change question
    documents = TEMP(question, 5)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState) -> Dict[str, Any]:

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    Web_Search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
    if len(filtered_docs) == 0:
        Web_Search=True
    return {"documents": filtered_docs, "question": question, "Web_Search": Web_Search}