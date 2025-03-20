import streamlit as st
import ollama

from src.utils import *
from src.model import *


# INITIALISATIONS
database_path = ""
if "model" not in st.session_state:
    st.session_state["model"] = "gemma:2b"   # MODELE

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "good_answers" not in st.session_state:
    st.session_state["good_answers"] = 0
if "css" not in st.session_state:
    style_path = "src/frontend/assets/style.css"
    with open(style_path) as f:
        css = f.read()
    st.session_state["css"] = css
if "logo" not in st.session_state:
    logo = "src/frontend/assets/bot.png"
    st.session_state["logo"] = logo


st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)
st.sidebar.image("src/frontend/assets/logo.png", use_container_width=True)


st.header("Chatbot")
st.logo(st.session_state["logo"])



def model_res_generator():
    """
    Generates a response from the model based on chat history
    """
    stream = ollama.chat(
        model=st.session_state["model"],    # MODELE USED HERE
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def get_context_prompt(prompt):
    """
    Enhances the user's message with context
    """
    return prompt

def evaluate_answer(answer):
    """
    Evaluates weither the user's answer is right or not then adds one two the counter of right answers
    """
    if True:
        st.session_state["good_answers"] += 1


# category selection option
category = st.selectbox(
    "Select a specific category to train on :",
    ("All categories", "c1", "c2"),
)
st.write("Category selected:", category)

st.divider()


# display message history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# get user message
if prompt := st.chat_input("Que voulez-vous savoir?"):
    context_prompt = get_context_prompt(prompt)

    # adding user message in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    # displaying newest message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # get and display model message
    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})

        # test for statistics variables (good answer or not)
        evaluate_answer(context_prompt)
