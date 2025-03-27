import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# initialisations
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
if 'clicked_demo' not in st.session_state:
    st.session_state.clicked_demo = False
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
st.sidebar.image("src/frontend/assets/logo.png", use_container_width=True)


st.header("Statistics")
st.logo(st.session_state.bot)



st.write("This page gathers all your answer statistics !")

st.divider()

col1, col2 = st.columns(2)

with col1:
    # RADAR CHART
    categories = list(st.session_state.category_scores.keys()) # category names
    scores = [st.session_state.category_scores[c] for c in categories] # number answers for each category
    nb_answers = max(max(scores), 5) # number of total given answers = questions

    # for the connecting line to be closed on the chart
    categories += [categories[0]]
    scores += [scores[0]]

    colors = [
        "rgb(245, 224, 220)",
        "rgb(242, 205, 205)",
        "rgb(245, 194, 231)",
        "rgb(203, 166, 247)",
        "rgb(243, 139, 168)",
        "rgb(235, 160, 172)",
        "rgb(250, 179, 135)",
        "rgb(249, 226, 175)",
        "rgb(166, 227, 161)",
        "rgb(148, 226, 213)",
        "rgb(137, 220, 235)",
        "rgb(116, 199, 236)",
        "rgb(137, 180, 250)",
        "rgb(180, 190, 254)"
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='tonext',
        fillcolor='rgba(242, 205, 205,0.7)',
        line=dict(color='rgb(49, 0, 85)', width=2),  # line to connect points
        marker=dict(color=colors[:len(categories)], size=10)  # one color for each point
    ))
    fig.update_layout(
        polar=dict(
            bgcolor = "rgba(242, 205, 205,0)",
            radialaxis=dict(
                visible=True,
                range=[0, nb_answers],
                showticklabels=True,
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            )
        ),
        showlegend=False,
        title=dict(
            text="Amount of answer per category",
            yref='paper',
            font=dict(size=18)
        ),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)




with col2:
    labels = ["Right answers","Wrong answers"]
    nb_questions = st.session_state.number_questions
    nb_bonnes_reponses = st.session_state.right_answers

    fig2 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig2.add_trace(go.Pie(
        labels=labels,
        values=[nb_bonnes_reponses, nb_questions - nb_bonnes_reponses],
        name="Answers",
        hole=.7,
        marker=dict(colors=['rgb(166, 227, 161)', 'rgba(0,0,0,0)']),  # colored for right answers, invisible for wrong ones
        hoverinfo="skip",
        showlegend=False,
        textinfo='none'
    ))

    fig2.update_layout(
        title_text="Global right answers rate",
        annotations=[dict(text=f'{nb_bonnes_reponses}/{nb_questions}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig2)

def click_button_demo():
    st.session_state.clicked_demo = True

_, center, _ = st.columns(3)
with center:
    if st.button('Demo score'):
        for category in st.session_state.category_scores.keys():
            st.session_state.nb_questions = 50
            rd_nb_question = np.random.randint(1,6)
            rd_score = np.random.randint(0,rd_nb_question)
            st.session_state.category_scores[category] += rd_nb_question
            st.session_state.right_answers += rd_score
            st.session_state.number_questions = sum([st.session_state.category_scores[category] for category in st.session_state.category_scores.keys()])
