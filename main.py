"""
Identity Building through questions
"""

import logging

import prompts
from llm_wrapper import send_llm_request
from models import ConversationResponse, Summary

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_new_topic_question(profile: str = "") -> str:
    """Generate a question about a completely new topic using LLM"""
    prompt = prompts.NEW_TOPIC_PROMPT
    if profile:
        prompt += prompts.NEW_TOPIC_WITH_PROFILE_PROMPT.format(profile=profile)

    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=prompts.SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )
    return response.question.strip()


def process_response(user_response: str, current_profile: str) -> ConversationResponse:
    """Process user response through LLM to update profile"""
    prompt = prompts.PROCESS_RESPONSE_PROMPT.format(
        current_profile=current_profile,
        user_response=user_response
    )
    return send_llm_request(
        model="gpt-4o-mini",
        system_prompt=prompts.SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )


def save_profile(narrative: str, filename: str = "profile.txt") -> None:
    """Save the user profile narrative to a text file"""
    # Generate and save summary if narrative is long enough
    if len(narrative.splitlines()) > 5:
        logger.debug("Profile too long, generating summary before saving...")
        narrative = generate_summary(narrative)
        logger.debug(f"Saving summarized profile:\n{narrative}")
    else:
        logger.debug(f"Saving profile:\n{narrative}")

    with open(filename, "w") as f:
        f.write(narrative)


def load_profile(filename: str = "profile.txt") -> str:
    """Load the user profile narrative from a text file"""
    try:
        with open(filename) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def generate_summary(profile: str) -> str:
    """Generate a one-sentence summary of everything we know"""
    if not profile:
        return ""

    prompt = prompts.SUMMARY_PROMPT.format(profile=profile)

    try:
        response = send_llm_request(
            model="gpt-4o-mini",
            system_prompt=prompts.SUMMARY_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=Summary,
            images=[],
        )

        if not response or not response.summary:
            logger.debug("No summary returned from LLM")
            return profile

        summary = response.summary.strip()
        logger.debug(f"Generated new summary: {summary}")

        # Validate the summary is actually shorter
        if len(summary.splitlines()) < len(profile.splitlines()):
            return summary
        else:
            logger.debug("Summary was not shorter than profile")
            return profile

    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return profile


def generate_initial_question(profile: str) -> str:
    """Generate a contextual opening question based on existing profile"""
    if not profile:
        return "Hi! Tell me something about yourself?"

    prompt = prompts.INITIAL_QUESTION_PROMPT.format(profile=profile)

    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=prompts.SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )
    return response.question


if __name__ == "__main__":
    profile = load_profile()

    if profile:
        logger.debug("Attempting to generate summary...")
        summary = generate_summary(profile)
        logger.debug(f"Generated summary: {summary}")
        if summary:
            profile = summary
            logger.debug("Profile replaced with summary")
            # Immediately save the summarized profile
            save_profile(profile)
            logger.debug("Saved summarized profile to file")
        else:
            logger.debug("Summary generation failed")

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
            profile = f'{profile}\n{new_profile}' if profile else new_profile

        question = generate_new_topic_question(profile)
        exchange_count += 1
        save_profile(profile)
