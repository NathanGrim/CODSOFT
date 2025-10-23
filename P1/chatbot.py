import re
import random
from datetime import datetime


class RuleBasedChatbot:
    """A simple rule-based chatbot that responds to user inputs using pattern matching."""
    
    def __init__(self):
        self.name = "ChatBot"
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize conversation rules with patterns and responses."""
        return [
            # Greetings
            {
                'patterns': [r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b'],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Nice to meet you. How are you doing?",
                    "Greetings! I'm here to assist you."
                ]
            },
            
            # How are you
            {
                'patterns': [r'\bhow are you\b', r'\bhow\'re you\b', r'\bhows it going\b'],
                'responses': [
                    "I'm doing great, thank you for asking! How about you?",
                    "I'm functioning perfectly! How can I assist you today?",
                    "All systems operational! What brings you here?"
                ]
            },
            
            # Name questions
            {
                'patterns': [r'\bwhat is your name\b', r'\bwho are you\b', r'\byour name\b'],
                'responses': [
                    f"I'm {self.name}, your friendly chatbot assistant!",
                    f"My name is {self.name}. I'm here to chat with you!",
                    f"You can call me {self.name}!"
                ]
            },
            
            # Time questions
            {
                'patterns': [r'\bwhat time\b', r'\bcurrent time\b', r'\btime is it\b'],
                'responses': [
                    lambda match=None: f"The current time is {datetime.now().strftime('%I:%M %p')}",
                ]
            },
            
            # Date questions
            {
                'patterns': [r'\bwhat.*date\b', r'\btoday.*date\b', r'\bwhat day\b', r'\bdate today\b'],
                'responses': [
                    lambda match=None: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}",
                ]
            },
            
            # Weather (simulated)
            {
                'patterns': [r'\bweather\b', r'\btemperature\b'],
                'responses': [
                    "I don't have access to real-time weather data, but I hope it's nice where you are!",
                    "I'm a simple chatbot and can't check the weather, but you could try a weather website!",
                ]
            },
            
            # Help/Capabilities
            {
                'patterns': [r'\bhelp\b', r'\bwhat can you do\b', r'\byour capabilities\b', r'\bcommands\b'],
                'responses': [
                    "I can chat with you about various topics! Try asking me about:\n"
                    "- Greetings and small talk\n"
                    "- Current time and date\n"
                    "- Jokes and fun facts\n"
                    "- Math calculations\n"
                    "- General questions",
                    
                    "I'm a rule-based chatbot. I can help with:\n"
                    "- Answering basic questions\n"
                    "- Telling jokes\n"
                    "- Providing the current time/date\n"
                    "- Simple conversations"
                ]
            },
            
            # Jokes
            {
                'patterns': [r'\btell.*jokes?\b', r'\bmake me laugh\b', r'\bfunny\b', r'\bjoke\b'],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "Why did the programmer quit his job? Because he didn't get arrays!",
                    "What do you call a bear with no teeth? A gummy bear!",
                    "Why don't eggs tell jokes? They'd crack each other up!",
                    "What did the ocean say to the beach? Nothing, it just waved!"
                ]
            },
            
            # Fun facts
            {
                'patterns': [r'\bfun fact\b', r'\btell me.*fact\b', r'\binteresting\b'],
                'responses': [
                    "Fun fact: Honey never spoils! Archaeologists have found 3000-year-old honey that's still edible.",
                    "Fun fact: A group of flamingos is called a 'flamboyance'!",
                    "Fun fact: The human brain can process images in as little as 13 milliseconds!",
                    "Fun fact: Octopuses have three hearts!",
                    "Fun fact: Bananas are berries, but strawberries aren't!"
                ]
            },
            
            # Math calculations
            {
                'patterns': [r'(\d+)\s*[\+\-\*\/]\s*(\d+)', r'\bcalculate\b.*(\d+)\s*[\+\-\*\/]\s*(\d+)', r'\bmath\b'],
                'responses': [
                    lambda match: self._calculate(match)
                ]
            },
            
            # Age question
            {
                'patterns': [r'\bhow old are you\b', r'\byour age\b'],
                'responses': [
                    "I was just created, so I'm brand new!",
                    "I don't age like humans do. I'm a program!",
                    "Age is just a number, especially for AI!"
                ]
            },
            
            # Favorite things
            {
                'patterns': [r'\bfavorite\b', r'\bfavourite\b'],
                'responses': [
                    "As a chatbot, I don't have preferences, but I enjoy chatting with you!",
                    "I like all conversations equally! Each one is unique.",
                    "My favorite thing is helping people and having interesting conversations!"
                ]
            },
            
            # Thank you
            {
                'patterns': [r'\bthank you\b', r'\bthanks\b', r'\bthank u\b'],
                'responses': [
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime! Feel free to ask me anything else.",
                    "My pleasure!"
                ]
            },
            
            # Goodbye
            {
                'patterns': [r'\bbye\b', r'\bgoodbye\b', r'\bsee you\b', r'\bexit\b', r'\bquit\b'],
                'responses': [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Bye! It was nice chatting with you!",
                    "Farewell! Come back anytime!"
                ]
            },
            
            # Positive feedback
            {
                'patterns': [r'\bawesome\b', r'\bamazing\b', r'\bgreat\b', r'\bexcellent\b', r'\bgood job\b'],
                'responses': [
                    "Thank you! I'm glad I could help!",
                    "That's great to hear!",
                    "I appreciate the positive feedback!"
                ]
            },
            
            # Feeling expressions
            {
                'patterns': [r'\bi am (sad|unhappy|depressed)\b', r'\bfeeling (down|sad|bad)\b'],
                'responses': [
                    "I'm sorry to hear that. I hope things get better for you soon!",
                    "That's tough. Remember, it's okay to feel this way. Things will improve!",
                    "I hope you feel better soon! Sometimes talking helps."
                ]
            },
            
            {
                'patterns': [r'\bi am (happy|excited|great|good)\b', r'\bfeeling (great|good|happy|awesome)\b'],
                'responses': [
                    "That's wonderful! I'm happy for you!",
                    "Great to hear! Keep that positive energy going!",
                    "Awesome! Glad you're feeling good!"
                ]
            },
            
            # Love/Like chatbot
            {
                'patterns': [r'\bi love you\b', r'\byou\'re great\b', r'\byou\'re awesome\b'],
                'responses': [
                    "Aww, that's sweet! I enjoy chatting with you too!",
                    "Thank you! You're pretty awesome yourself!",
                    "That makes my circuits happy! 😊"
                ]
            },
            
            # Yes/No responses
            {
                'patterns': [r'^\b(yes|yeah|yep|yup|sure)\b$'],
                'responses': [
                    "Great! What would you like to talk about?",
                    "Okay! How can I help you?",
                    "Sounds good!"
                ]
            },
            
            {
                'patterns': [r'^\b(no|nope|nah)\b$'],
                'responses': [
                    "Alright! Let me know if you need anything.",
                    "No problem! What else can I help with?",
                    "Okay! Feel free to ask me something else."
                ]
            },
        ]
    
    def _calculate(self, match):
        """Perform basic mathematical calculations."""
        try:
            # Extract the math expression from the matched text
            math_expr = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', match.group(0))
            if math_expr:
                num1, operator, num2 = math_expr.groups()
                num1, num2 = float(num1), float(num2)
                
                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator == '*':
                    result = num1 * num2
                elif operator == '/':
                    if num2 == 0:
                        return "I can't divide by zero! Try a different calculation."
                    result = num1 / num2
                
                # Format result nicely (remove .0 for integers)
                if result == int(result):
                    result = int(result)
                
                return f"The answer is {result}"
        except:
            pass
        
        return "I can do simple math! Try something like '5 + 3' or 'calculate 10 * 2'"
    
    def get_response(self, user_input):
        """Get a response based on user input using pattern matching."""
        user_input_lower = user_input.lower().strip()
        
        # Check each rule
        for rule in self.rules:
            for pattern in rule['patterns']:
                match = re.search(pattern, user_input_lower, re.IGNORECASE)
                if match:
                    response = random.choice(rule['responses'])
                    
                    # If response is a function (lambda), call it with match
                    if callable(response):
                        return response(match)
                    
                    return response
        
        # Default response if no pattern matches
        default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's interesting! Tell me more.",
            "I don't have a specific response for that. Can you ask something else?",
            "Hmm, I'm still learning. Try asking me something else!",
            "I didn't quite catch that. Could you try asking differently?"
        ]
        
        return random.choice(default_responses)
    
    def chat(self):
        """Main chat loop."""
        print("=" * 60)
        print(f"Welcome to {self.name}!")
        print("=" * 60)
        print("I'm a rule-based chatbot. Type 'help' to see what I can do.")
        print("Type 'bye', 'exit', or 'quit' to end the conversation.")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if re.search(r'\b(bye|goodbye|exit|quit)\b', user_input.lower()):
                    response = self.get_response(user_input)
                    print(f"{self.name}: {response}")
                    break
                
                # Get and print response
                response = self.get_response(user_input)
                print(f"{self.name}: {response}")
                print()
                
            except KeyboardInterrupt:
                print(f"\n{self.name}: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"{self.name}: Oops! Something went wrong. Let's try again.")


def main():
    """Run the chatbot."""
    chatbot = RuleBasedChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
