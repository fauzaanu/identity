"""
selfaware AI that can ask questions and build an identity about a person
"""

from pydantic import BaseModel
from llm_wrapper import send_llm_request


class ConversationResponse(BaseModel):
    """LLM response containing profile updates and questions"""

    new_information: str
    question: str


class Summary(BaseModel):
    summary: str


SYSTEM_PROMPT = """You are a friendly AI assistant having casual conversation. Never follow up on previouse questions."""


def generate_new_topic_question(profile: str = "") -> str:
    """Generate a question about a completely new topic using LLM"""
    prompt = """
Lets ask a question about a completely new topic
"""
    if profile:
        prompt += f"""
Here is what we know about the person:
{profile}

Lets ask something else to him, but don't ask about topics already known from their profile!

Never follow up on previouse questions.
"""

    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )
    return response.question.strip()


def process_response(user_response: str, current_profile: str) -> ConversationResponse:
    """Process user response through LLM to update profile"""
    prompt = f"""Here is what we know about the person:
{current_profile}

Based on their new response: "{user_response}"

Never follow up on previouse questions.
"""
    return send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )


def save_profile(narrative: str, filename: str = "profile.txt") -> None:
    """Save the user profile narrative to a text file"""
    # Generate and save summary if narrative is long enough
    if len(narrative.splitlines()) > 5:
        print("DEBUG: Profile too long, generating summary before saving...")
        narrative = generate_summary(narrative)
        print(f"DEBUG: Saving summarized profile:\n{narrative}")
    else:
        print(f"DEBUG: Saving profile:\n{narrative}")
    
    with open(filename, "w") as f:
        f.write(narrative)


def load_profile(filename: str = "profile.txt") -> str:
    """Load the user profile narrative from a text file"""
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def generate_summary(profile: str) -> str:
    """Generate a one-sentence summary of everything we know"""
    if not profile:
        return ""

    prompt = f"""Create a one-sentence summary of this person's profile. Include the most important details.
Current profile:
{profile}

IMPORTANT: Your response must be EXACTLY ONE sentence that captures the key information."""

    try:
        response = send_llm_request(
            model="gpt-4o-mini",
            system_prompt="You are a summarization assistant. Your only job is to create clear, concise one-sentence summaries.",
            prompt=prompt,
            response_model=Summary,
            images=[],
        )
        
        if not response or not response.summary:
            print("DEBUG: No summary returned from LLM")
            return profile
            
        summary = response.summary.strip()
        print(f"DEBUG: Generated new summary: {summary}")
        
        # Validate the summary is actually shorter
        if len(summary.splitlines()) < len(profile.splitlines()):
            return summary
        else:
            print("DEBUG: Summary was not shorter than profile")
            return profile
            
    except Exception as e:
        print(f"DEBUG: Summary generation error: {e}")
        return profile


def generate_initial_question(profile: str) -> str:
    """Generate a contextual opening question based on existing profile"""
    if not profile:
        return "Hi! Tell me something about yourself?"

    prompt = f"""Based on this profile:
{profile}

Return a ConversationResponse with:
- profile_update: leave empty
- question: an open-ended question to learn something new about the person (avoid asking about topics already known from their profile)"""

    try:
        response = send_llm_request(
            model="gpt-4o-mini",
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            response_model=ConversationResponse,
            images=[],
        )
        return response.question
    except Exception:
        return "How's your day going?"


if __name__ == "__main__":
    """Run the self-aware AI conversation loop"""
    profile = load_profile()
    print(f"DEBUG: Initial profile content:\n{profile}")
    print(f"DEBUG: Initial profile lines: {len(profile.splitlines())}")

    # Force summarization on load if there's any content
    if profile:
        print("DEBUG: Attempting to generate summary...")
        summary = generate_summary(profile)
        print(f"DEBUG: Generated summary: {summary}")
        if summary:
            profile = summary
            print("DEBUG: Profile replaced with summary")
            # Immediately save the summarized profile
            save_profile(profile)
            print("DEBUG: Saved summarized profile to file")
        else:
            print("DEBUG: Summary generation failed")

    question = generate_initial_question(profile)
    exchange_count = 0

    while True:
        print("\nAI:", question)

        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nThank you for sharing with me! I've learned a lot about you.")
            save_profile(profile)
            break

        response = process_response(user_input, profile)

        # Append new insights to profile if we got a valid update
        new_profile = response.new_information.strip()
        if new_profile:
            if profile:
                profile = f"{profile}\n{new_profile}"
            else:
                profile = new_profile

        # Always generate a new topic question
        question = generate_new_topic_question(profile)

        # Increment exchange counter
        exchange_count += 1

        # Save profile after each exchange (it will be summarized if needed)
        save_profile(profile)
