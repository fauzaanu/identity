"""
selfaware AI that can ask questions and build an identity about a person
"""
import json
from datetime import datetime
from pydantic import BaseModel

from llm_wrapper import send_llm_request

class Profile(BaseModel):
    """Stores a narrative description of the person"""
    narrative: str
    last_updated: datetime

class ConversationResponse(BaseModel):
    """LLM response containing profile updates"""
    profile_update: str

SYSTEM_PROMPT = """You are a friendly AI assistant having casual conversations to learn about people.
Keep questions light, short, and easy to answer.
Focus on everyday topics like hobbies, interests, and preferences.
Avoid heavy or personal topics unless the person brings them up first."""

def generate_new_topic_question() -> str:
    """Generate a question about a completely new topic using LLM"""
    prompt = """Ask a short, casual question about their interests or daily life.
Keep it light and easy to answer in a sentence or two."""


    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )
    return response.profile_update.strip()

def process_response(user_response: str, current_profile: Profile) -> ConversationResponse:
    """Process user response through LLM to update profile"""
    prompt = f"""Current profile of the person:
{current_profile.narrative}

Based on their new response: "{user_response}"

Update and expand the profile narrative to incorporate any new insights about their identity."""
    return send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )

def save_profile(profile: Profile, filename: str = "profile.json") -> None:
    """Save the user profile to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(profile.model_dump(), f, default=str, indent=2)

def load_profile(filename: str = "profile.json") -> Profile:
    """Load the user profile from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return Profile(**data)
    except FileNotFoundError:
        return Profile(
            narrative="",
            last_updated=datetime.now()
        )

def generate_initial_question(profile: Profile) -> str:
    """Generate a contextual opening question based on existing profile"""
    if not profile.narrative:
        return "Hi! What do you like to do for fun?"

    prompt = f"""Based on what I know about the person:
{profile.narrative}

Generate a single engaging opening question that shows awareness of their profile while seeking new information.
Make it natural and conversational."""

    try:
        response = send_llm_request(
            model="gpt-4o-mini",
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            response_model=ConversationResponse,
            images=[],
        )
        return response.follow_up_question
    except Exception:
        return "How's your day going?"

def main():
    """Run the self-aware AI conversation loop"""
    profile = load_profile()

    print("Hello! I'm your AI companion, and I remember what I know about you.")
    question = generate_initial_question(profile)

    while True:
        print("\nAI:", question)

        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for sharing with me! I've learned a lot about you.")
            save_profile(profile)
            break

        try:
            response = process_response(user_input, profile)

            # Update profile with new insights
            profile.narrative = response.profile_update
            profile.last_updated = datetime.now()

            # Generate a completely new topic question
            question = generate_new_topic_question()

            # Save profile after each exchange
            save_profile(profile)

        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
