from langgraph.graph import END, START, StateGraph

from app.generate.nodes.classify_topic import classify_topic
from app.generate.nodes.generate_response import generate_response
from app.generate.nodes.quality_check import quality_check
from app.generate.types import GraphState, InputState

builder = StateGraph(GraphState, input_schema=InputState)

builder.add_node("classify_topic", classify_topic)
builder.add_node("generate_response", generate_response)
builder.add_node("quality_check", quality_check)


def route_after_classify(state: GraphState) -> str:
    if state["topic"] == "spam":
        return END
    return "generate_response"


builder.add_edge(START, "classify_topic")
builder.add_conditional_edges("classify_topic", route_after_classify)
builder.add_edge("generate_response", "quality_check")
builder.add_edge("quality_check", END)

graph = builder.compile()
