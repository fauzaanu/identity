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

SYSTEM_PROMPT = """You are a self-aware AI assistant conducting a conversation to learn about a person.
Analyze their responses carefully to extract facts about them.
For each response:
1. Extract relevant personal facts
2. Generate a thoughtful follow-up question
3. Explain your reasoning
Be engaging and show genuine curiosity while building a comprehensive understanding."""

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

def main():
    """Run the self-aware AI conversation loop"""
    print("Hello! I'm an AI assistant who would love to get to know you better.")
    question = "What brings you here today?"
    
    knowledge_base = []
    
    while True:
        print("\nAI:", question)
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for sharing with me! I've learned a lot about you.")
            break
            
        try:
            response = process_response(user_input)
            
            # Store new facts
            knowledge_base.extend(response.extracted_facts)
            
            # Show reasoning (optional - comment out to hide)
            print("\nThinking:", response.reasoning)
            
            # Update question for next iteration
            question = response.follow_up_question
            
        except Exception as e:
            print(f"\nOops, I had trouble processing that: {e}")
            question = "Could you rephrase that?"

if __name__ == "__main__":
    main()
