"""
selfaware AI that can ask questions and build an identity about a person
"""
from pydantic import BaseModel
from llm_wrapper import send_llm_request

class ConversationResponse(BaseModel):
    """LLM response containing profile updates and questions"""
    profile_update: str
    question: str

SYSTEM_PROMPT = """You are a friendly AI assistant having casual conversations to learn about people.
Keep all responses extremely brief and direct.
When updating the profile, use simple factual statements.
No analysis or elaboration - just state the facts in 1-2 short sentences."""

def generate_new_topic_question() -> str:
    """Generate a question about a completely new topic using LLM"""
    prompt = """Generate a question for the user.
Return a ConversationResponse with:
- profile_update: leave empty
- question: a short, casual question about their interests or daily life"""


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
    prompt = f"""Current profile of the person:
{current_profile}

Based on their new response: "{user_response}"

Return a ConversationResponse with:
- profile_update: a very brief update incorporating their response (1-2 simple factual statements)
- question: a natural follow-up question about what they just shared"""
    return send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )

def save_profile(narrative: str, filename: str = "profile.txt") -> None:
    """Save the user profile narrative to a text file"""
    with open(filename, 'w') as f:
        f.write(narrative)

def load_profile(filename: str = "profile.txt") -> str:
    """Load the user profile narrative from a text file"""
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def generate_initial_question(profile: str) -> str:
    """Generate a contextual opening question based on existing profile"""
    if not profile:
        return "Hi! What do you like to do for fun?"

    prompt = f"""Based on what I know about the person:
{profile}

Return a ConversationResponse with:
- profile_update: leave empty
- question: a single engaging question that shows awareness of their profile"""

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

            # Update profile with new insights if we got a valid update
            new_profile = response.profile_update.strip()
            if new_profile:
                profile = new_profile

            # Use the follow-up question from the response
            question = response.question
            if not question:
                # Only generate a new topic if we didn't get a follow-up question
                question = generate_new_topic_question() or "Could you tell me more about yourself?"

            # Save profile after each exchange
            save_profile(profile)

        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
