import streamlit as st
import ollama
import numpy as np
import re 
import os

from src.utils import *
from src.model import *

# INITIALISATIONS
database_path = ""
if "model" not in st.session_state:
    st.session_state.model = "gemma3:4b"   # MODELE

if "messages" not in st.session_state:
    system_prompt = 'Tu es un assistant de révision utile pour les étudiants en droit et spécialisé dans les brevets intellectuels. Tu es pédagogue et plein de ressources. Ton ojectif est de poser des questions à l\'étudiant sans y répondre puis de les corriger en étant explicite de si sa réponse est juste ou non en citant les sources précisemment. Tes questions vont droit au but, respectent le type et la catégorie données par l\'étudiant et ne sont pas superflues. Le format des réponses que tu fournis pour les questions de type QCM est le suivant : (A) Réponse A, (B) Réponse B, etc.'
    st.session_state.messages = [{'role': 'system', 'content': system_prompt}]
if "good_answers" not in st.session_state:
    st.session_state.good_answers = 0
if 'clicked_question' not in st.session_state:
    st.session_state.clicked_question = False
if "app_name" not in st.session_state:
    app_name =  "Better Call X"
    st.session_state.app_name = app_name
if "css" not in st.session_state:
    style_path = "src/frontend/assets/style.css"
    with open(style_path) as f:
        css = f.read()
    st.session_state.css = css
if "bot" not in st.session_state:
    bot = "src/frontend/assets/bot.png"
    st.session_state.bot = bot

st.set_page_config(page_title=st.session_state.app_name, page_icon=st.session_state.bot)
st.markdown(f'<style>{st.session_state.css}</style>', unsafe_allow_html=True)
st.sidebar.image("src/frontend/assets/logo.png", use_container_width=True)


st.header("Chatbot")
st.logo(st.session_state.bot)

def model_res_generator():
    """
    Generates a response from the model based on chat history
    """
    stream = ollama.chat(
        model=st.session_state.model,    # MODELE USED HERE
        messages=st.session_state.messages,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def parse_qcm_response(response):
    """
    Extracts QCM choices from the model response, supports formats: (A), A), A. and A-
    """
    choices = re.findall(r"(?:\([A-Z]\)|[A-Z]\)|[A-Z]\.|[A-Z]-)\s.*", response)
    return [choice.strip() for choice in choices]

def get_context_prompt(prompt):
    """
    Enhances the user's message with context
    """
    return prompt

def click_button_question():
    st.session_state.clicked_question = True

def evaluate_answer(answer):
    """
    Evaluates weither the user's answer is right or not then adds one two the counter of right answers
    """
    if True:
        st.session_state.good_answers += 1

def generate_question(type, category):
    """
    Returns a prompt for a question with the right type and category
    """
    return f"Pose-moi une question de type {type} sur la catégorie {category} sans y répondre."

def display_qcm_choices():
    """
    Displays QCM choices if they are stored in session state
    """
    if "qcm_choices" in st.session_state and st.session_state["qcm_choices"]:
        selected_choice = st.radio(
            "Choisissez votre réponse :",
            st.session_state["qcm_choices"],
            key="qcm_response"
        )
        if st.button("Valider la réponse"):
            reponse_utilisateur = f"Réponse choisie : {selected_choice}"
            # Réinitialiser les choix après soumission
            st.session_state["qcm_choices"] = []

            st.session_state["messages"].append({"role": "user", "content": reponse_utilisateur})
            with st.chat_message("user"):
                st.markdown(reponse_utilisateur)
                
            with st.chat_message("assistant"):
                message = st.write_stream(model_res_generator())
                st.session_state["messages"].append({"role": "assistant", "content": message})


# question category and type selection
right, left = st.columns(2,gap='large')
with right: 
    category = st.selectbox(
        "Choisissez une catégorie sur laquelle vous entraîner :",
        ("Toutes les catégories", "chat", "dragon"),
    )
    st.write("Catégorie sélectionnée :", category)
with left:
    st.radio(
            "Sélectionnez le type de question",
            ["QCM", "Ouverte", "Peu importe"],
            key="type_question",
            horizontal=True,
        )
    if st.session_state.type_question == "Peu importe":
        random = np.random.random()
        if random < 0.5:
            type = "QCM"
        else:
            type = "Ouverte"
    else:
        type = st.session_state.type_question

_, center, _ = st.columns(3)
with center:
    st.button('Pose-moi une question !', on_click=click_button_question)

st.divider()


# display message history

for message in st.session_state["messages"]:
    if message['role'] != 'system':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# get user message
prompt = st.chat_input("Que voulez-vous savoir?")
if prompt:
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


if st.session_state.clicked_question:
    context_prompt = generate_question(type, category)

    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    with st.chat_message("user"):
        st.markdown(context_prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})

        qcm_choices = parse_qcm_response(message)
        if qcm_choices:
            st.session_state["qcm_choices"] = qcm_choices
    
    st.session_state.clicked_question = False

display_qcm_choices()

print(st.session_state.messages)