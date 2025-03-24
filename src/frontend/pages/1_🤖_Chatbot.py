import streamlit as st
import ollama
import numpy as np
import re 

from src.utils import *
from src.model import *

# INITIALISATIONS
database_path = ""
if "model" not in st.session_state:
    st.session_state.model = "gemma3:4b"   # MODELE
if "g" not in st.session_state:
    st.session_state.g = 'graph'
if "messages" not in st.session_state:
    system_prompt = 'Tu es un assistant de révision pour les étudiants en droit et spécialisé dans les brevets intellectuels. ' \
    'Tu es pédagogue et plein de ressources. Ton ojectif est de poser des questions à l\'étudiant sans y répondre en t\'appuyant sur les documents à ta disposition. ' \
    'Lorsque tu génères une question de type QCM, le format des réponses que tu fournis est le suivant : (A) Réponse A, (B) Réponse B, etc.' \
    'Lorsque tu corriges la réponse donnée par un étudiant, indique explicitement "correcte" ou "incorrecte" en premier mot. Si "incorrecte", donne la bonne réponse parmi les propositions. Puis détaille les raisons en t\'appuyant sur des sources tirées des documents. ' \
    'Tes questions vont droit au but, respectent le type et la catégorie donnés par l\'étudiant et n\'ont pas de texte superflus.'
    st.session_state.messages = [{'role': 'system', 'content': system_prompt}]
if 'clicked_question' not in st.session_state:
    st.session_state.clicked_question = False
if "answers" not in st.session_state:
    st.session_state.answers = 0
if 'category_scores' not in st.session_state: # right answers counter for each category
    st.session_state.category_scores = {
        "Amendements et octroi": 0,
        "Biotechnologies et listes de séquences": 0,
        "Demandes de priorité et droit de priorité": 0,
        "Demandes divisionnaires": 0,
        "Droits et transferts": 0,
        "Droit substantiel des brevets : nouveauté et activité inventive": 0,
        "Examen": 0,
        "Exigences et formalités de dépôt": 0,
        "Langues et traductions": 0,
        "Oppositions et appels": 0,
        "Procédure PCT et entrée dans la phase européenne": 0,
        "Recours procéduraux et effet juridique": 0,
        "Taxes, méthodes de paiement et délais": 0,
        "Unité de l'invention": 0
    }
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

def real_number_of_questions():
    count=0
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            if "Correcte." or "Incorrecte." in message["content"]:
                count += 1
    return count

def evaluate_answer(answer, category):
    """
    Evaluates weither the user's answer is right or not then adds one two the counter category right answers
    """
    if st.session_state.answers < real_number_of_questions(): # avoid adding more for each page load
        if "Correcte." in answer:
            st.session_state.category_scores[category] += 1
            st.session_state.answers += 1
        elif "Incorrecte." in answer:
            st.session_state.answers += 1
        # else : model answer was not evaluating a response

def generate_question(type, category):
    """
    Returns a prompt for a question with the right type and category
    """
    return f"Pose-moi une question de type {type} sur la catégorie {category}."

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

def choice_between(list):
    choice = str( np.random.choice(list))
    return choice

def linkify_articles(response, g):
    """
    Transforms a text to another by replacing "Article X" to an hyperlink to the article 
    """
    def replace_article(match):
        article_number = match.group(1)
        article_name = f"Article {article_number}"
        if article_name in g.nodes:
            try:
                url = g.nodes[article_name]["data"].metadata["url"]
                return f"[{article_name}]({url})"
            except KeyError:
                pass
        # returns original title if not foudn
        return article_name

    linked_response = re.sub(r"Article (\d+)", replace_article, response)
    return linked_response


# question category and type selection
right, left = st.columns(2,gap='large')
with right: 
    category = st.selectbox(
        "Choisissez une catégorie sur laquelle vous entraîner :",
        ("Toutes les catégories", "Amendements et octroi", "Biotechnologies et listes de séquences", "Demandes de priorité et droit de priorité", "Demandes divisionnaires","Droits et transferts","Droit substantiel des brevets : nouveauté et activité inventive","Examen","Exigences et formalités de dépôt","Langues et traductions","Oppositions et appels","Procédure PCT et entrée dans la phase européenne","Recours procéduraux et effet juridique","Taxes, méthodes de paiement et délais","Unité de l'invention"),
    )
    if category == "Toutes les catégories":
        category = choice_between(["Amendements et octroi", "Biotechnologies et listes de séquences", "Demandes de priorité et droit de priorité", "Demandes divisionnaires","Droits et transferts","Droit substantiel des brevets : nouveauté et activité inventive","Examen","Exigences et formalités de dépôt","Langues et traductions","Oppositions et appels","Procédure PCT et entrée dans la phase européenne","Recours procéduraux et effet juridique","Taxes, méthodes de paiement et délais","Unité de l'invention"])
with left:
    st.radio(
            "Sélectionnez le type de question",
            ["QCM", "Ouverte", "Peu importe"],
            key="type_question",
            horizontal=True,
        )
    if st.session_state.type_question == "Peu importe":
        type = choice_between(['QCM', 'Ouverte'])
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
        #linked_message = linkify_articles(message, st.session_state.g)
        st.session_state["messages"].append({"role": "assistant", "content": message})

if st.session_state.clicked_question:
    context_prompt = generate_question(type, category)

    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    with st.chat_message("user"):
        st.markdown(context_prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        #linked_message = linkify_articles(message, st.session_state.g)
        st.session_state["messages"].append({"role": "assistant", "content": message})  # remember to switch to linked_message here

        qcm_choices = parse_qcm_response(message)
        if qcm_choices:
            st.session_state["qcm_choices"] = qcm_choices
    
    st.session_state.clicked_question = False

display_qcm_choices()
if st.session_state.messages[-1]["role"] == "assistant":
    evaluate_answer(st.session_state.messages[-1]["content"],category)
