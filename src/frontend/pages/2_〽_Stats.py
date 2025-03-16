import streamlit as st

st.set_page_config(page_title="Application", page_icon="📚")
st.header("Statistiques")

st.write("On pourra mettre des stats sur le pourcentage de bonnes réponses ou autre ")

st.markdown(st.session_state["good_answers"])