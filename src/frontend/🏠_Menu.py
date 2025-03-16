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
if "css" not in st.session_state:
    style_path = "src/frontend/assets/style.css"
    with open(style_path) as f:
        css = f.read()
    st.session_state["css"] = css
if "logo" not in st.session_state:
    logo = "src/frontend/assets/cat.jpg"
    st.session_state["logo"] = logo

st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)
st.logo(st.session_state["logo"])



st.markdown(
    """
    ## Bienvenue sur la page d'accueil
    coucou
"""
)



# Les variables peuvent être passées d'une page à l'autre avec le st.session_state

