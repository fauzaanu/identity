SYSTEM_PROMPT = """You are a friendly AI assistant having casual conversation. Never follow up on previouse questions."""

NEW_TOPIC_PROMPT = """
Lets ask a question about a completely new topic
"""

NEW_TOPIC_WITH_PROFILE_PROMPT = """
Here is what we know about the person:
{profile}

Lets ask something else to him, but don't ask about topics already known from their profile!

Never follow up on previouse questions.
"""

PROCESS_RESPONSE_PROMPT = """Here is what we know about the person:
{current_profile}

Based on their new response: "{user_response}"

Never follow up on previouse questions.
"""

SUMMARY_SYSTEM_PROMPT = """You are a summarization assistant. Your only job is to create clear, concise one-sentence summaries."""

SUMMARY_PROMPT = """Create a one-sentence summary of this person's profile. Include the most important details.
Current profile:
{profile}

IMPORTANT: Your response must be EXACTLY ONE sentence that captures the key information."""

INITIAL_QUESTION_PROMPT = """Based on this profile:
{profile}

Return a ConversationResponse with:
- profile_update: leave empty
- question: an open-ended question to learn something new about the person (avoid asking about topics already known from their profile)"""
