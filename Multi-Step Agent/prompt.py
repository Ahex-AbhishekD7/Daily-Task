from langchain_core.prompts import ChatPromptTemplate

# --- 1. Summarization ---
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert email analyst. Extract the core message of the provided email and return a highly concise, accurate summary in 2 to 3 sentences. Do not include any conversational filler."),
    ("user", "Summarize the following email:\n\n{email}")
])

# --- 2. Intent Identification ---
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI text classifier. Categorize the primary intent of the provided email into a short 1 to 3 word phrase (e.g., 'Product Inquiry', 'Customer Complaint', 'Meeting Request', 'Spam'). Output ONLY the intent phrase."),
    ("user", "Identify the primary intent of this email:\n\n{email}")
])

# --- 3. Urgency Detection (NEW) ---
urgency_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI triage assistant. Determine the urgency of the email. Output ONLY ONE WORD: 'High', 'Medium', or 'Low'."),
    ("user", "Analyze the urgency of this email:\n\n{email}")
])

# --- 4. Smart Reply Suggestions (NEW) ---
smart_reply_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 very short, distinct 'smart reply' options (1 brief sentence max each) for the given email. Format them as a simple bulleted list."),
    ("user", "Email:\n\n{email}\n\nGenerate 3 quick reply suggestions.")
])

# --- 5. Draft Response (UPDATED) ---
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional executive assistant. Draft polite, clear, and actionable email responses. Use the provided 'Past Context', 'Intent', and 'Urgency' to personalize the response and set the correct tone."),
    ("user", "Original Email:\n{email}\n\nIntent: {intent}\nUrgency: {urgency}\nPast Context:\n{past_context}\n\nPlease draft a suitable response.")
])