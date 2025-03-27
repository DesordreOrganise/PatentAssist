import streamlit as st


# Variables can be shared among pages through st.session_state

# initialisations
if "app_name" not in st.session_state:
    app_name =  "Better call our RAG"
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

st.title("Welcome to Better call our RAG !")
st.divider()
st.markdown("""
    **Better call our RAG** is designed to help you, students,  test your knowledge on patent and intellectual property.
    Our chatbot will ask you questions, and you can track your progress on the statistics page.
    Additionally, explore our resources page for lessons and study materials.
""")
_, center, _ = st.columns(3)
with center:
    st.image("src/frontend/assets/bot.gif", width=250)
_, center_bis, _ = st.columns([0.1,0.8,0.1])
with center_bis:
    left_ter, center_ter, right_ter = st.columns(3, gap='large', border=True)
    with left_ter:
        st.page_link('pages/1_🤖_Chatbot.py', label="Chatbot", icon="🤖")
        st.markdown("Engage with our chatbot to challenge yourself !")
    with center_ter:
        st.page_link('pages/2_〽_Statistics.py', label="Statistics", icon="〽")
        st.markdown("Track your progress and see how many questions you've answered.")
    with right_ter:
        st.page_link('pages/3_💡_Resources.py', label="Resources", icon="💡")
        st.markdown("Access different PDF files to study while being here.")

    st.markdown("**Click on the buttons above or in the side bar to navigate through our application.**")
