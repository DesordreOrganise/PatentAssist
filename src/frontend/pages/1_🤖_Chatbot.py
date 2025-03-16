import streamlit as st
import ollama

from src.utils import *
from src.model import *

st.set_page_config(page_title="Application", page_icon="📚")



st.header("Chatbot")
# chargement la base de données
database_path = ""

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = "gemma:2b"

def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


# affichage de l'historique des messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Que voulez-vous savoir?"):
    context_prompt = prompt

    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": context_prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        # pour gérer la mémoire, ajouter ici le contenu du message de l'assistant avec le bon role
        st.session_state["messages"].append({"role": "assistant", "content": message})
