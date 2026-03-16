# AI Agent

A small terminal support agent I built using LangGraph. You type a question, it figures out what you need, and either answers it, opens a ticket, or gets a human on the line. No UI, no external APIs — just Python and your terminal.

---

## What's inside

The graph has two nodes — `reason` and `respond`. That's it.

`reason` reads the user's message, decides which tool fits, runs it, and drops the result into the message history. `respond` picks that result up and hands it back as the reply.

The three tools are plain functions:

| Tool | Does |
|---|---|
| `knowledge_search` | checks a local dict for common questions |
| `create_ticket` | makes a `TKT-XXXX` id for anything unknown |
| `escalate` | passes the user to a human agent |

A `memory` dict lives in the state and counts turns through the session.

---

## Getting started

> You'll need Python 3.9 or newer.

```bash
pip install -r requirements.txt
python Architecture.py
```

---

## Quick demo

```
Support Agent  (type 'quit' to exit)

Agent: Hi! What can I help you with today?

You: how do I reset my password?

Agent: Head to Settings → Security → Reset Password and we'll email you a link.
[memory] turns=1

You: I'd like to talk to someone

Agent: Connecting you with Sarah — should be about 3–5 minutes.
[memory] turns=2

You: quit
Agent: Take care!
```
