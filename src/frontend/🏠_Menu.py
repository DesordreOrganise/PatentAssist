import streamlit as st


# Variables can be shared among pages through st.session_state

# initialisations
if "app_name" not in st.session_state:
    app_name =  "Better Call X"
    st.session_state["app_name"] = app_name
if "css" not in st.session_state:
    style_path = "src/frontend/assets/style.css"
    with open(style_path) as f:
        css = f.read()
    st.session_state["css"] = css
if "bot" not in st.session_state:
    bot = "src/frontend/assets/bot.png"
    st.session_state["bot"] = bot

st.set_page_config(page_title=st.session_state.app_name, page_icon=st.session_state.bot)
st.markdown(f'<style>{st.session_state.css}</style>', unsafe_allow_html=True)
st.logo(st.session_state.bot)

st.sidebar.image("src/frontend/assets/logo.png", width=250)

st.title("Bienvenue sur Better Call X !")

_, center, _ = st.columns(3)
with center:
    st.image("src/frontend/assets/bot.png", width=300)
    st.markdown("Discutez avec notre chatbot pour tester vos connaissances sur les brevets intelectuels !")
    
    _, center_bis, _ = st.columns(3)
    with center_bis:
        st.page_link('pages/1_🤖_Chatbot.py', label="Chatbot", icon="🤖")
    
    st.markdown("Cliquez sur que le bouton ci-dessus ou dans la barre latérale pour accéder au chat.")




