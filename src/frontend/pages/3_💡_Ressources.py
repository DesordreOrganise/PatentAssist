import streamlit as st
import base64

st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)

st.header("Ressources PDF")
st.logo(st.session_state["logo"])

st.markdown("Cliquer sur la 🡆 pour visualiser le PDF.")

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

pdfs = [
    "resources/cours_pdf/1-EPC_17th_edition_2020_en.pdf",
    "resources/cours_pdf/2-PCT_wipo-pub-274-2024-en-patent-cooperation-treaty.pdf",
    "resources/cours_pdf/3-en-epc-guidelines-2024-hyperlinked.pdf",
    "resources/cours_pdf/case_law_of_the_boards_of_appeal_2022_en.pdf",
    "resources/cours_pdf/en-pct-epo-guidelines-2024-hyperlinked.pdf",
    "resources/cours_pdf/en-up-guidelines-2025-pre-publication.pdf"
]

left, right = st.columns([0.3, 0.7], gap='large')

with left:
    for pdf in pdfs:
        pdf_name = pdf.split("/")[-1]
        st.write(pdf_name)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            display_file(pdf)
        with col2:
            if st.button("🡆", key=pdf):
                with right:
                    display_pdf(pdf)