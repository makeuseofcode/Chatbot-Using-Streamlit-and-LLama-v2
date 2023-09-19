import streamlit as st
import os
from utils import debounce_replicate_run

# External libraries
import replicate

# Set initial page configuration
st.set_page_config(
    page_title="LLaMA2Chat",
    page_icon=":volleyball:",
    layout="wide"
)

# Global variables
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', default='')

# Define model endpoints as independent variables
LLaMA2_7B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')
LLaMA2_13B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')
LLaMA2_70B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')

PRE_PROMPT = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."

import streamlit as st

# Constants
LLaMA2_MODELS = {
    'LLaMA2-7B': LLaMA2_7B_ENDPOINT,
    'LLaMA2-13B': LLaMA2_13B_ENDPOINT,
    'LLaMA2-70B': LLaMA2_70B_ENDPOINT,
}

# Session State Variables
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_PRE_PROMPT = PRE_PROMPT

def setup_session_state():
    st.session_state.setdefault('chat_dialogue', [])
    selected_model = st.sidebar.selectbox('Choose a LLaMA2 model:', list(LLaMA2_MODELS.keys()), key='model')
    st.session_state.setdefault('llm', LLaMA2_MODELS.get(selected_model, LLaMA2_70B_ENDPOINT))
    st.session_state.setdefault('temperature', DEFAULT_TEMPERATURE)
    st.session_state.setdefault('top_p', DEFAULT_TOP_P)
    st.session_state.setdefault('max_seq_len', DEFAULT_MAX_SEQ_LEN)
    st.session_state.setdefault('pre_prompt', DEFAULT_PRE_PROMPT)

def render_sidebar():
    st.sidebar.header("LLaMA2 Chatbot")
    st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=DEFAULT_TEMPERATURE, step=0.01)
    st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=DEFAULT_TOP_P, step=0.01)
    st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=4096, value=DEFAULT_MAX_SEQ_LEN, step=8)
    new_prompt = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', DEFAULT_PRE_PROMPT, height=60)
    if new_prompt != DEFAULT_PRE_PROMPT and new_prompt != "" and new_prompt is not None:
        st.session_state['pre_prompt'] = new_prompt + "\n\n"
    else:
        st.session_state['pre_prompt'] = DEFAULT_PRE_PROMPT

def render_chat_history():
    response_container = st.container()
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    user_input = st.chat_input("Type your question here to talk to LLaMA2")
    if user_input:
        st.session_state.chat_dialogue.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

def generate_assistant_response():
    message_placeholder = st.empty()
    full_response = ""
    string_dialogue = st.session_state['pre_prompt']
    
    for dict_message in st.session_state.chat_dialogue:
        speaker = "User" if dict_message["role"] == "user" else "Assistant"
        string_dialogue += f"{speaker}: {dict_message['content']}\n\n"
    
    output = debounce_replicate_run(
        st.session_state['llm'],
        string_dialogue + "Assistant: ",
        st.session_state['max_seq_len'],
        st.session_state['temperature'],
        st.session_state['top_p'],
        REPLICATE_API_TOKEN
    )
    
    for item in output:
        full_response += item
        message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    st.session_state.chat_dialogue.append({"role": "assistant", "content": full_response})

def render_app():
    setup_session_state()
    render_sidebar()
    render_chat_history()
    handle_user_input()
    generate_assistant_response()

def main():
    render_app()

if __name__ == "__main__":
    main()

