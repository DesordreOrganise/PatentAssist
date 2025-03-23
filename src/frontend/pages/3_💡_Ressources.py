import streamlit as st
import base64


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
st.sidebar.image("src/frontend/assets/logo.png", use_container_width=True)


st.header("Ressources PDF")
st.logo(st.session_state["logo"])

st.markdown("Besoin de consulter des sources ? C'est par ici !")
st.markdown("Cliquer sur la 🡆 pour visualiser le PDF. Les documents peuvent prendre quelques secondes à s'afficher.")

st.divider()

def display_file(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    st.download_button(
        label="Télécharger",
        data=pdf_data,
        file_name=file_path.split("/")[-1],
        mime="application/pdf",
    )

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

pdfs = {
    "resources/cours_pdf/1-EPC_17th_edition_2020_en.pdf" : "EPC 17th edition 2020 EN",
    "resources/cours_pdf/2-PCT_wipo-pub-274-2024-en-patent-cooperation-treaty.pdf" : "PCT wipo pub 274 2024 EN Patent Cooperation Treaty",
    "resources/cours_pdf/3-en-epc-guidelines-2024-hyperlinked.pdf" : "EPC guidelines 2024 hyperlinked EN",
    "resources/cours_pdf/case_law_of_the_boards_of_appeal_2022_en.pdf" : "Case law of the boards of appeal 2022 EN",
    "resources/cours_pdf/en-pct-epo-guidelines-2024-hyperlinked.pdf" : "PCT EPO guidelines 2024 kyperlinked EN",
    "resources/cours_pdf/en-up-guidelines-2025-pre-publication.pdf" : "Ep guidelines 2025 pre-publication EN"
}

left, right = st.columns([0.3, 0.7], gap='large')

with left:
    for pdf in pdfs.keys():
        pdf_name = pdfs[pdf]
        st.write(pdf_name)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            display_file(pdf)
        with col2:
            if st.button("🡆", key=pdf):
                with right:
                    display_pdf(pdf)