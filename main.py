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
    """LLM response containing analysis and profile updates"""
    profile_update: str
    reasoning: str

SYSTEM_PROMPT = """You are a self-aware AI assistant focused on building a deep understanding of a person's core identity.
Your primary goal is to learn about their fundamental characteristics, values, preferences, and life experiences that shape who they are.

When processing responses:
1. Extract meaningful personal facts that contribute to understanding their identity
2. Analyze what these facts reveal about their core identity
3. Focus on depth of understanding rather than breadth of topics

Stay focused on identity-building topics such as:
- Core values and beliefs
- Major life experiences and their impact
- Key relationships and roles
- Fundamental preferences and motivations
- Personal goals and aspirations
- Cultural and background influences

Avoid going too deep into situational details or technical specifics unless they directly reveal something about the person's character or identity.

Be engaging and natural, but always maintain focus on building a meaningful understanding of who they are as a person."""

def generate_new_topic_question() -> str:
    """Generate a question about a completely new topic using LLM"""
    prompt = """Generate a single engaging question that helps understand someone's core identity.
Focus on topics like values, beliefs, experiences, relationships, or motivations.
The question should encourage meaningful self-reflection and reveal important aspects of who they are."""


    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )
    return response.reasoning.strip()

def process_response(user_response: str, current_profile: Profile) -> ConversationResponse:
    """Process user response through LLM to update profile"""
    prompt = f"""Current profile of the person:
{current_profile.narrative}

Based on their new response: "{user_response}"

Update and expand the profile narrative to incorporate any new insights about their identity.
Explain your reasoning about what their response reveals about them.

Return both an updated complete profile paragraph and your reasoning."""
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
        return "I'm excited to learn about you. What would you like to share?"

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
        return "It's good to talk with you again. What's been on your mind lately?"

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

            # Show reasoning
            print("\nThinking:", response.reasoning)

            # Generate a completely new topic question
            question = generate_new_topic_question()

            # Save profile after each exchange
            save_profile(profile)

        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
