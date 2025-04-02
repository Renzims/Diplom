from langgraph.graph import END, StateGraph
from typing import Any, Dict
from state import GraphState
from grader import retrieval_grader
from nodes import retrieve,grade_documents,decide_to_generate

GRADE_DOCUMENTS="GRADE_DOCUMENTS"
RETRIEVE="RETRIEVE"
Decide_to_generate="Decide_to_generate"



workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.set_entry_point(RETRIEVE)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(Decide_to_generate, decide_to_generate)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(GRADE_DOCUMENTS, Decide_to_generate)

app = workflow.compile()

if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")