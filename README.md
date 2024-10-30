# Identity Building AI Assistant

An AI-powered conversational agent that builds a detailed understanding of a person through natural dialogue. The assistant asks contextual questions, maintains a profile of learned information, and generates summaries of what it knows about the user.

## Features

- Natural conversation flow with contextually aware questions
- Maintains and updates user profile based on responses
- Automatic profile summarization
- Avoids repetitive topics by tracking previous conversations
- Proper logging for debugging and monitoring

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main script:
```bash
python main.py
```

The assistant will:
1. Load any existing profile (if present)
2. Start a conversation with contextual questions
3. Save and update the profile as you chat
4. Generate summaries automatically when needed

To exit the conversation, type: "quit", "exit", or "bye"

## Project Structure

- `main.py`: Core conversation logic and profile management
- `models.py`: Pydantic models for LLM responses
- `llm_wrapper.py`: OpenAI API integration
- `prompts.py`: System and conversation prompts

## License

MIT License
