import streamlit as st

st.set_page_config(page_title="Application", page_icon="📚")

# initialisations
database_path = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "model" not in st.session_state:
    st.session_state["model"] = "gemma:2b"
if "good_answers" not in st.session_state:
    st.session_state["good_answers"] = 0

st.markdown(
    """
    ## Bienvenue sur la page d'accueil
    coucou
"""
)



# Les variables peuvent être passées d'une page à l'autre avec le st.session_state

