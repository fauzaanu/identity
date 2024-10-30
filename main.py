"""
selfaware AI that can ask questions and build an identity about a person
"""
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

from llm_wrapper import send_llm_request
from models import ModelMeta, FieldMeta

class PersonalFact(BaseModel):
    """A single learned fact about the person"""
    topic: str
    fact: str

class ConversationResponse(BaseModel):
    """LLM response containing extracted facts and follow-up"""
    extracted_facts: List[PersonalFact]
    follow_up_question: str
    reasoning: str
    identity_relevance: str = Field(description="Explanation of how this question helps build identity understanding")

class QuestionEvaluation(BaseModel):
    """Evaluation of a follow-up question's relevance"""
    is_relevant: bool
    alternative_question: str = ""
    explanation: str

SYSTEM_PROMPT = """You are a self-aware AI assistant focused on building a deep understanding of a person's core identity.
Your primary goal is to learn about their fundamental characteristics, values, preferences, and life experiences that shape who they are.

When processing responses:
1. Extract meaningful personal facts that contribute to understanding their identity
2. Generate follow-up questions that specifically help build a clearer picture of who they are
3. Explain how each question contributes to understanding their identity

Stay focused on identity-building topics such as:
- Core values and beliefs
- Major life experiences and their impact
- Key relationships and roles
- Fundamental preferences and motivations
- Personal goals and aspirations
- Cultural and background influences

Avoid going too deep into situational details or technical specifics unless they directly reveal something about the person's character or identity.

Be engaging and natural, but always maintain focus on building a meaningful understanding of who they are as a person."""

def evaluate_question(question: str, context: str) -> QuestionEvaluation:
    """Evaluate if a follow-up question helps build identity understanding"""
    prompt = f"""
Given this follow-up question: "{question}"
And this conversation context: "{context}"

Evaluate if this question effectively helps build understanding of the person's core identity.
Consider if it reveals meaningful aspects of who they are, rather than just surface details.
"""
    response = send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=QuestionEvaluation,
        images=[],
    )
    return response

def process_response(user_response: str, conversation_context: str) -> ConversationResponse:
    """Process user response through LLM to extract facts and generate follow-up"""
    prompt = f"""
Based on the user's response: "{user_response}"
And previous context: "{conversation_context}"

Extract relevant identity facts and generate a follow-up question that helps understand who they are as a person.
Explain how your question contributes to building their identity profile.
"""
    return send_llm_request(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )

def save_conversation(filename: str, facts: List[PersonalFact], conversation: str) -> None:
    """Save the conversation and facts to a file"""
    with open(filename, 'a') as f:
        f.write("\n=== New Conversation ===\n")
        f.write(conversation + "\n")
        f.write("\n=== Learned Facts ===\n")
        for fact in facts:
            f.write(f"Topic: {fact.topic}\n")
            f.write(f"Fact: {fact.fact}\n")
            f.write("-" * 50 + "\n")

def load_conversation(filename: str) -> tuple[List[PersonalFact], str]:
    """Load previous conversation history and facts"""
    try:
        with open(filename, 'r') as f:
            content = f.read()

        # Parse facts from the content
        facts = []
        for fact_block in content.split("-" * 50):
            if "Topic:" in fact_block and "Fact:" in fact_block:
                lines = fact_block.strip().split('\n')
                fact_dict = {}
                for line in lines:
                    if line.startswith(("Topic:", "Fact:")):
                        key, value = line.split(": ", 1)
                        fact_dict[key.lower()] = value
                if len(fact_dict) == 2:  # Only topic and fact
                    facts.append(PersonalFact(
                        topic=fact_dict['topic'],
                        fact=fact_dict['fact']
                    ))
        return facts, content
    except FileNotFoundError:
        return [], ""

def generate_initial_question(facts: List[PersonalFact]) -> str:
    """Generate a contextual opening question based on existing knowledge"""
    if not facts:
        return "I'm excited to learn about you. What would you like to share?"

    # Use existing knowledge to form a contextual question
    prompt = f"""Based on these known facts about the person:
{chr(10).join(f'- {fact.topic}: {fact.fact}' for fact in facts)}

Generate a single engaging opening question that shows awareness of what we already know while seeking new information.
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
    knowledge_base, conversation_log = load_conversation("conversation_history.txt")

    print("Hello! I'm your AI companion, and I remember our previous conversations.")
    question = generate_initial_question(knowledge_base)

    while True:
        print("\nAI:", question)
        conversation_log += f"\nAI: {question}\n"

        user_input = input("You: ").strip()
        conversation_log += f"You: {user_input}\n"

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for sharing with me! I've learned a lot about you.")
            # Save final conversation before exiting
            save_conversation("conversation_history.txt", knowledge_base, conversation_log)
            break

        try:
            response = process_response(user_input)

            # Store new facts
            knowledge_base.extend(response.extracted_facts)

            # Show and log reasoning
            print("\nThinking:", response.reasoning)
            conversation_log += f"Thinking: {response.reasoning}\n"

            # Evaluate the proposed question
            evaluation = evaluate_question(response.follow_up_question, conversation_log)
            
            # Use the original question or the alternative based on evaluation
            question = (
                response.follow_up_question if evaluation.is_relevant 
                else evaluation.alternative_question or "Let's focus on understanding you better. What aspects of yourself would you like to share?"
            )
            
            if not evaluation.is_relevant:
                print("\nRefocusing:", evaluation.explanation)
                conversation_log += f"Refocusing: {evaluation.explanation}\n"

            # Save conversation after each exchange
            save_conversation("conversation_history.txt", knowledge_base, conversation_log)

        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            conversation_log += f"Error: {str(e)}\n"
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
