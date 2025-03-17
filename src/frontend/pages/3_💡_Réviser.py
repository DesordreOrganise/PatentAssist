import streamlit as st

st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)

st.header("Réviser")
st.logo(st.session_state["logo"])

