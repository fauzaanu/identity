from pydantic import BaseModel


class ConversationResponse(BaseModel):
    """LLM response containing profile updates and questions"""

    new_information: str
    question: str


class Summary(BaseModel):
    summary: str
