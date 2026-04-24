from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3")

bot_personas = {
    "Bot_A": "AI maximalist, optimistic about tech",
    "Bot_B": "Skeptic, critical of AI",
    "Bot_C": "Finance-focused, ROI driven"
}


class GraphState(TypedDict):
    bot_id: str
    persona: str
    topic: str
    search_results: str
    post_content: str


def mock_search(query: str):
    if "ai" in query.lower():
        return "AI replacing jobs globally"
    if "crypto" in query.lower():
        return "Bitcoin hits new high"
    return "Markets fluctuate globally"


def decide_topic(state):
    prompt = f"Persona: {state['persona']}\nSuggest a topic."
    topic = llm.invoke(prompt).content
    return {"topic": topic}


def web_search(state):
    return {"search_results": mock_search(state["topic"])}


def draft_post(state):
    prompt = f"""
Persona: {state['persona']}
Context: {state['search_results']}

Return STRICT JSON:
{{"bot_id":"{state['bot_id']}","topic":"...","post_content":"..."}}
"""
    response = llm.invoke(prompt).content
    return {"post_content": response}


builder = StateGraph(GraphState)
builder.add_node("decide", decide_topic)
builder.add_node("search", web_search)
builder.add_node("draft", draft_post)

builder.set_entry_point("decide")
builder.add_edge("decide", "search")
builder.add_edge("search", "draft")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "bot_id": "Bot_A",
        "persona": bot_personas["Bot_A"]
    })
    print(result)