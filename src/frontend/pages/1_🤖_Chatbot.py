import streamlit as st
import ollama
import numpy as np
import re

# this because it crash otherwise
import torch
import os
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from src.system.rag import RAG
from src.utils.stream_tools import default_rag


# INITIALISATIONS
database_path = ""
if "model" not in st.session_state:
    st.session_state.model = "gemma3:4b"   # MODELE

if "g" not in st.session_state or "rag" not in st.session_state:
    rag, g= default_rag()
    print("RAG and graph loaded")
    st.session_state.g = g
    print("Graph loaded\n\n\n\n\n")
    st.session_state.rag = rag
    print("RAG loaded\n\n\n\n\n")

if "category_2_idx" not in st.session_state:
    st.session_state.category_2_idx = {
        "Filing requirements and formalities": 1,
        "Priority claims and right of priority": 2,
        "Divisional applications": 3,
        "Fees, payment methods, and time limits": 4,
        "Languages and translations": 5,
        "Procedural remedies and legal effect": 6,
        "Pct procedure and entry into the european phase": 7,
        "Examination, amendments, and grant": 8,
        "Opposition and appeals": 9,
        "Substantive patent law: novelty and inventive step": 10,
        "Entitlement and transfers": 11,
        "Biotech and sequence listings": 12,
        "Unity of invention": 13
    }


if "messages" not in st.session_state: #"""relying on the documents at your disposal."""
    system_prompt = 'You are a revision assistant for law students specializing in patent and intellectual property.  ' \
    'You are pedagogical and resourceful. Your goal is to ask the student questions by generating them and without answering them, ' \
    'When you generate a MCQ, the format of the answers you provide is as follows: (A) Answer A, (B) Answer B, etc. ' \
    'When you correct a student\'s answer, explicitly state "Correct." or "Incorrect." as the first word. If "incorrect," provide the correct answer from the options. Then detail the reasons among sources.' \
    'Your questions are straightforward, adhere to the type and category specified by the student, and contain no superfluous text.'
    st.session_state.messages = [{'role': 'system', 'content': system_prompt}]

if 'clicked_question' not in st.session_state:
    st.session_state.clicked_question = False

if "right_answers" not in st.session_state:
    st.session_state.right_answers = 0

if "number_questions" not in st.session_state:
    st.session_state.number_questions = 0

if 'category_scores' not in st.session_state: # answers counter for each category
    st.session_state.category_scores = {
        "Biotech and sequence listings": 0,
        "Divisional applications": 0,
        "Entitlement and transfers": 0,
        "Examination, amendments, and grant": 0,
        "Fees, payment methods, and time limits": 0,
        "Filing requirements and formalities": 0,
        "Languages and translations": 0,
        "Opposition and appeals": 0,
        "Pct procedure and entry into the european phase": 0,
        "Priority claims and right of priority": 0,
        "Procedural remedies and legal effect": 0,
        "Substantive patent law: novelty and inventive step": 0,
        "Unity of invention": 0
    }

if "app_name" not in st.session_state:
    app_name =  "Better call our RAG"
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
    # stream = ollama.chat(
    #     model=st.session_state.model,    # MODELE USED HERE
    #     messages=st.session_state.messages,
    #     stream=True,
    # )
    # for chunk in stream:
    #     yield chunk["message"]["content"]
    last_user_message = st.session_state["messages"][-1]["content"]
    print("last user message : ", last_user_message)
    for chunk in st.session_state.rag.run_flux(last_user_message):
        yield chunk

def parse_qcm_response(response):
    """
    Extracts QCM choices from the model response, supports formats: (A), A), A. and A-
    """
    choices = re.findall(r"(?:\([A-Z]\)|[A-Z]\.|[A-Z]-)\s.*", response)
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
            if "Correct." or "Incorrect." in message["content"]:
                count += 1
    return count

def evaluate_answer(answer, category):
    """
    Evaluates weither the user's answer is right or not then adds one two the counter category and to the irght answers global if needed
    """
    if st.session_state.number_questions < real_number_of_questions(): # avoid adding more for each page load
        if "Correct." in answer:
            st.session_state.category_scores[category] += 1
            st.session_state.right_answers += 1
            st.session_state.number_questions += 1
        elif "Incorrect." in answer:
            st.session_state.category_scores[category] += 1
            st.session_state.number_questions += 1
        # else : model answer was not evaluating a response

def generate_question(type, category):
    """
    Returns a prompt for a question with the right type and category
    """
    return f"Ask me a {type} question about the category {category}."

def ask_for_hint():
    print("Coucou l'indice !")

def display_qcm_choices():
    """
    Displays QCM choices if they are stored in session state
    """
    if "qcm_choices" in st.session_state and st.session_state["qcm_choices"]:
        selected_choice = st.radio(
            "Choose your answer :",
            st.session_state["qcm_choices"],
            key="qcm_response"
        )
        if st.button("Give me a hint !"):
            ask_for_hint()
        if st.button("Send"):
            reponse_utilisateur = f"Chosen answer : {selected_choice}"
            st.session_state["qcm_choices"] = []

            st.session_state["messages"].append({"role": "user", "content": reponse_utilisateur})
            with st.chat_message("user"):
                st.markdown(reponse_utilisateur)
                
            with st.chat_message("assistant"):
                message = st.write_stream(st.session_state.rag.run_flux(reponse_utilisateur))
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
        "Choose a category to train on :",
        ("All categories", "Biotech and sequence listings","Divisional applications","Entitlement and transfers","Examination, amendments, and grant","Fees, payment methods, and time limits","Filing requirements and formalities","Languages and translations","Opposition and appeals","Pct procedure and entry into the european phase","Priority claims and right of priority","Procedural remedies and legal effect","Substantive patent law: novelty and inventive step","Unity of invention")
    )
    if category == "All categories":
        category = choice_between(["Biotech and sequence listings","Divisional applications","Entitlement and transfers","Examination, amendments, and grant","Fees, payment methods, and time limits","Filing requirements and formalities","Languages and translations","Opposition and appeals","Pct procedure and entry into the european phase","Priority claims and right of priority","Procedural remedies and legal effect","Substantive patent law: novelty and inventive step","Unity of invention"])
with left:
    st.radio(
            "Select the question type",
            ["MCQ", "Open", "Any type"],
            key="type_question",
            horizontal=True,
        )
    if st.session_state.type_question == "Any type":
        type = choice_between(['MCQ', 'Open'])
    else:
        type = st.session_state.type_question

_, center, _ = st.columns(3)
with center:
    st.button('Ask me a question !', on_click=click_button_question)

st.divider()


# display message history
for message in st.session_state["messages"]:
    if message['role'] != 'system':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# get user message
prompt = st.chat_input("What do you want to know?")
if prompt:
    context_prompt = get_context_prompt(prompt)

    # adding user message in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    # displaying newest message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # get and display model message
    with st.chat_message("assistant"):
        message = st.write_stream(st.session_state.rag.run_flux(context_prompt))
        #linked_message = linkify_articles(message, st.session_state.g)     # HERE TO GET HYPERLINKS
        st.session_state["messages"].append({"role": "assistant", "content": message})

if st.session_state.clicked_question:
    print("clicked question")
    print(str(st.session_state.category_2_idx[category]) + '. ' + category)
    context_prompt = generate_question(type, category)

    st.session_state["messages"].append({"role": "user", "content": context_prompt})
    with st.chat_message("user"):
        st.markdown(context_prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(st.session_state.rag.generate_question(str(st.session_state.category_2_idx[category]) + '. ' + category, type=type)) # generate_question
        linked_message = linkify_articles(message, st.session_state.g)   # HERE TO GET HYPERLINKS
        st.session_state["messages"].append({"role": "assistant", "content": message})  # remember to switch to linked_message here

        qcm_choices = parse_qcm_response(message)
        if qcm_choices:
            st.session_state["qcm_choices"] = qcm_choices
    
    st.session_state.clicked_question = False

display_qcm_choices()
if st.session_state.messages[-1]["role"] == "assistant":
    evaluate_answer(st.session_state.messages[-1]["content"],category)
