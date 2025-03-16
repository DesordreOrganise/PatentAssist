import streamlit as st


st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)

st.header("Statistiques")
st.logo(st.session_state["logo"])



st.write("On pourra mettre des stats sur le pourcentage de bonnes réponses ou autre ")

st.markdown(st.session_state["good_answers"])