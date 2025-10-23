# Rule-Based Chatbot

A simple rule-based chatbot that responds to user inputs using predefined rules and pattern matching. This project demonstrates basic natural language processing concepts and conversation flow.

## Features

- **Pattern Matching**: Uses regular expressions to identify user queries
- **Multiple Response Categories**:
  - Greetings and farewells
  - Time and date queries
  - Jokes and fun facts
  - Basic math calculations
  - Help and capabilities
  - Emotional responses
  - Small talk
- **Randomized Responses**: Provides varied responses to keep conversations natural
- **Error Handling**: Gracefully handles unexpected inputs
- **Easy to Extend**: Simple rule structure makes it easy to add new patterns and responses

## Requirements

- Python 3.6 or higher
- Streamlit (for web interface)

## Installation

1. Clone or download this repository
2. Install dependencies (only needed for web interface):

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Interface (Recommended) 🌐

Run the Streamlit web app for a beautiful, interactive interface:

```bash
python -m streamlit run app.py
```

Or alternatively:
```bash
streamlit run app.py
```

This will open a web browser with a modern chat interface featuring:
- 🎨 **Beautiful UI** with color-coded message bubbles
- 📜 **Chat History** - all your conversations saved
- ⏰ **Timestamps** on every message
- 🗑️ **Clear Chat** button to start fresh
- 📋 **Sidebar** with commands and information
- 📱 **Responsive Design** - works on any device

### Option 2: Command Line Interface 💻

Run the traditional command-line chatbot:

```bash
python chatbot.py
```

### Example Conversations

```
You: Hello!
ChatBot: Hi there! What can I do for you?

You: What's your name?
ChatBot: I'm ChatBot, your friendly chatbot assistant!

You: Tell me a joke
ChatBot: Why don't scientists trust atoms? Because they make up everything!

You: What time is it?
ChatBot: The current time is 02:30 PM

You: Calculate 15 + 7
ChatBot: The answer is 22.0

You: Help
ChatBot: I can chat with you about various topics! Try asking me about:
- Greetings and small talk
- Current time and date
- Jokes and fun facts
- Math calculations
- General questions

You: Bye
ChatBot: Goodbye! Have a great day!
```

## How It Works

The chatbot uses a rule-based approach with the following components:

1. **Pattern Recognition**: Regular expressions match user input patterns
2. **Response Selection**: Each matched pattern has associated responses
3. **Random Selection**: Responses are randomly selected for variety
4. **Lambda Functions**: Some responses use functions for dynamic content (time, calculations)
5. **Fallback Mechanism**: Default responses when no pattern matches

## Extending the Chatbot

To add new conversation rules, add entries to the `rules` list in the `_initialize_rules()` method:

```python
{
    'patterns': [r'\byour pattern here\b'],
    'responses': [
        "Response option 1",
        "Response option 2",
        "Response option 3"
    ]
}
```

### Pattern Tips

- Use `\b` for word boundaries (e.g., `\bhello\b` matches "hello" but not "hellothere")
- Use `.*` to match any characters (e.g., `\btell.*joke\b` matches "tell me a joke")
- Use `|` for alternatives (e.g., `\b(hi|hello|hey)\b`)
- Use `\d+` to match numbers

## Project Structure

```
p1/
├── app.py              # Streamlit web interface (recommended)
├── chatbot.py          # Core chatbot logic & CLI version
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Limitations

- No machine learning or AI understanding
- Responses based solely on pattern matching
- Cannot learn from conversations
- No memory of previous messages
- Limited to predefined rules

## Future Enhancements

- Add conversation history/context
- Implement more sophisticated NLP techniques
- Add sentiment analysis
- Include external API integrations (weather, news, etc.)
- Add logging and analytics
- Create a GUI interface

## License

This project is open source and available for educational purposes.

## Author

Created as part of a CodSoft internship project to demonstrate rule-based chatbot development.
