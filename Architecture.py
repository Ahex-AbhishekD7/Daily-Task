from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import random


# --- state ---

class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: dict


# --- tools ---

kb = {
    "password": "Head to Settings → Security → Reset Password and we'll email you a link.",
    "order":    "Orders usually land in 3–5 business days. You can track them under the Orders tab.",
    "refund":   "Once we get the return, refunds hit your account in about 5–7 business days.",
    "locked":   "Accounts lock after 5 bad attempts. Wait 30 min or use the email unlock link.",
    "billing":  "You can find invoices and update your card under Account → Billing.",
}

def knowledge_search(query: str) -> str:
    for key, answer in kb.items():
        if key in query.lower():
            return answer
    return "not_found"

def create_ticket(issue: str) -> str:
    tid = f"TKT-{random.randint(1000, 9999)}"
    return f"Opened ticket {tid} for your issue."

def escalate() -> str:
    agents = ["Sarah", "Mike", "Alex"]
    return f"Connecting you with {random.choice(agents)} — should be about 3–5 minutes."


# --- nodes ---

def reason(state: State) -> State:
    msg = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    q   = msg.lower()

    if any(w in q for w in ["human", "agent", "person", "speak", "escalate"]):
        result = escalate()
    else:
        result = knowledge_search(msg)
        if result == "not_found":
            result = create_ticket(msg)

    state["memory"]["turns"] = state["memory"].get("turns", 0) + 1

    tool_msg = ToolMessage(content=result, tool_call_id="call_1")
    return {"messages": [tool_msg], "memory": state["memory"]}


def respond(state: State) -> State:
    result = next(m.content for m in reversed(state["messages"]) if isinstance(m, ToolMessage))
    reply  = AIMessage(content=result)
    return {"messages": [reply], "memory": state["memory"]}


# --- graph ---

graph = StateGraph(State)
graph.add_node("reason", reason)
graph.add_node("respond", respond)
graph.set_entry_point("reason")
graph.add_edge("reason", "respond")
graph.add_edge("respond", END)
agent = graph.compile()


# --- run ---

def main():
    state = {"messages": [], "memory": {}}
    print("\nSupport Agent  (type 'quit' to exit)\n")
    print("Agent: Hi! What can I help you with today?\n")

    while True:
        user = input("You: ").strip()
        if not user or user.lower() == "quit":
            print("Agent: Take care!\n")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user)]
        state = agent.invoke(state)

        reply = next(m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        print(f"\nAgent: {reply}")
        print(f"[memory] turns={state['memory'].get('turns', 0)}\n")


if __name__ == "__main__":
    main()