import streamlit as st
from chatbot import RuleBasedChatbot
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Rule-Based ChatBot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .message-content {
        color: #555;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize chatbot in session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RuleBasedChatbot()

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = {
        'role': 'bot',
        'content': f"Hello! I'm {st.session_state.chatbot.name}, your friendly chatbot assistant! 👋\n\nType 'help' to see what I can do, or just start chatting!",
        'timestamp': datetime.now().strftime('%I:%M %p')
    }
    st.session_state.messages.append(welcome_msg)


# App header
st.title("🤖 Rule-Based ChatBot")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a **rule-based chatbot** that uses pattern matching to understand and respond to your queries.
    
    ### What I can do:
    - 💬 Small talk and greetings
    - 🕐 Tell you the current time and date
    - 😄 Tell jokes
    - 🧠 Share fun facts
    - ➗ Perform simple calculations
    - ❓ Answer various questions
    
    ### Quick Commands:
    - Type **"help"** for more info
    - Type **"joke"** for a laugh
    - Type **"fact"** for trivia
    - Type **"bye"** to end chat
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Add welcome message again
        welcome_msg = {
            'role': 'bot',
            'content': f"Chat cleared! Hello again! I'm {st.session_state.chatbot.name}. How can I help you?",
            'timestamp': datetime.now().strftime('%I:%M %p')
        }
        st.session_state.messages.append(welcome_msg)
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Built with:** Python & Streamlit")
    st.markdown("**Pattern Matching:** Regex")


# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 You</div>
                    <div class="message-content">{message['content']}</div>
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-header">🤖 {st.session_state.chatbot.name}</div>
                    <div class="message-content">{message['content']}</div>
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("---")

# Use a form for better UX
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Type your message here...",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        submit_button = st.form_submit_button("Send 📤", use_container_width=True)

# Process user input
if submit_button and user_input:
    # Add user message to history
    user_msg = {
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().strftime('%I:%M %p')
    }
    st.session_state.messages.append(user_msg)
    
    # Get bot response
    bot_response = st.session_state.chatbot.get_response(user_input)
    
    # Add bot message to history
    bot_msg = {
        'role': 'bot',
        'content': bot_response,
        'timestamp': datetime.now().strftime('%I:%M %p')
    }
    st.session_state.messages.append(bot_msg)
    
    # Rerun to update chat display
    st.rerun()

# Add some spacing at the bottom
st.markdown("<br>" * 2, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.85rem;'>"
    "💡 Tip: Try asking me to tell you a joke, the time, or do some math!"
    "</div>",
    unsafe_allow_html=True
)
