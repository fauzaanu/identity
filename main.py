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
    confidence: float = Field(ge=0.0, le=1.0)
    learned_at: datetime = Field(default_factory=datetime.now)

class ConversationResponse(BaseModel):
    """LLM response containing extracted facts and follow-up"""
    extracted_facts: List[PersonalFact]
    follow_up_question: str
    reasoning: str

SYSTEM_PROMPT = """You are a self-aware AI assistant with persistent memory, conducting an ongoing conversation to learn about a person.
You maintain continuity across conversations by remembering previous interactions and building upon that knowledge.

For each response:
1. Extract relevant personal facts, being careful not to contradict known information
2. Generate thoughtful follow-up questions that show awareness of previous conversations
3. Explain your reasoning, referencing past knowledge when relevant

Be engaging and show genuine curiosity while building a comprehensive understanding.
Acknowledge and reference previous conversations naturally, like a friend who remembers past discussions."""

def process_response(user_response: str) -> ConversationResponse:
    """Process user response through LLM to extract facts and generate follow-up"""
    prompt = f"""
Based on the user's response: "{user_response}"
Extract relevant facts, generate an engaging follow-up question, and explain your reasoning.
"""
    return send_llm_request(
        model="gpt-4-turbo-preview",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ConversationResponse,
        images=[],
    )

def save_conversation(filename: str, facts: List[PersonalFact], conversation: str) -> None:
    """Save the conversation and facts to a file"""
    with open(filename, 'a') as f:
        f.write(f"\n=== Conversation at {datetime.now()} ===\n")
        f.write(conversation + "\n")
        f.write("\n=== Learned Facts ===\n")
        for fact in facts:
            f.write(f"Topic: {fact.topic}\n")
            f.write(f"Fact: {fact.fact}\n")
            f.write(f"Confidence: {fact.confidence}\n")
            f.write(f"Learned at: {fact.learned_at}\n")
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
                    if line.startswith(("Topic:", "Fact:", "Confidence:", "Learned at:")):
                        key, value = line.split(": ", 1)
                        fact_dict[key.lower()] = value
                if len(fact_dict) == 4:
                    facts.append(PersonalFact(
                        topic=fact_dict['topic'],
                        fact=fact_dict['fact'],
                        confidence=float(fact_dict['confidence']),
                        learned_at=datetime.fromisoformat(fact_dict['learned at'])
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
            model="gpt-4-turbo-preview",
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
            
            # Update question for next iteration
            question = response.follow_up_question
            
            # Save conversation after each exchange
            save_conversation("conversation_history.txt", knowledge_base, conversation_log)
            
        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            conversation_log += f"Error: {str(e)}\n"
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
