import streamlit as st
import ollama

from src.utils import *
from src.model import *


st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)

st.header("Chatbot")
st.logo(st.session_state["logo"])



def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def get_context_prompt(prompt):
    return prompt

def evaluate_answer(answer):
    st.session_state["good_answers"] += 1



# affichage de l'historique des messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# récupération du message utilisateur
if prompt := st.chat_input("Que voulez-vous savoir?"):
    context_prompt = get_context_prompt(prompt)

    # on ajoute le message utilisateur au format {role, content}
    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    # affichage du nouveau message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # récupération et affichage réponse modèle
    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})
        evaluate_answer(context_prompt)
