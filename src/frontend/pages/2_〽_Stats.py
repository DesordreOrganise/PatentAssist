import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Application", page_icon="📚")
st.markdown(f'<style>{st.session_state["css"]}</style>', unsafe_allow_html=True)

st.header("Statistiques")
st.logo(st.session_state["logo"])



st.write("Cette page regroupe les statistiques liées à tes réponses !")

st.markdown(st.session_state["good_answers"])

st.divider()

col1, col2 = st.columns(2)

with col1:
    # RADAR CHART
    categories = ['c1','c2','c3','c4', 'c5'] # categories name
    scores = [4,8,2,6,7] # good answers for each category
    nb_questions = 15

    # for the connecting line to be closed on the chart
    categories += [categories[0]]
    scores += [scores[0]]

    colors = [
        'rgb(245, 194, 231)',
        'rgb(203, 166, 247)',
        'rgb(243, 139, 168)',
        'rgb(250, 179, 135)',
        'rgb(249, 226, 175)',
        'rgb(166, 227, 161)',
        'rgb(148, 226, 213)',
        'rgb(116, 199, 236)',
        'rgb(180, 190, 254)'
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='tonext',
        fillcolor='rgba(242, 205, 205,0.2)',
        line=dict(color='rgb(242, 205, 205)', width=2),  # line to connect points
        marker=dict(color=colors[:len(categories)], size=10)  # one color for each point
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, nb_questions],
                showticklabels=True,
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            )
        ),
        showlegend=False,
        title=dict(
            text="Total des bonnes réponses par catégorie",
            yref='paper',
            font=dict(size=18)
        ),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)




with col2:
    labels = ["Bonnes réponses","Mauvaises réponses"]
    nb_bonnes_reponses = 12

    fig2 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig2.add_trace(go.Pie(
        labels=labels,
        values=[nb_bonnes_reponses, nb_questions - nb_bonnes_reponses],
        name="Réponses",
        hole=.7,
        marker=dict(colors=['rgb(166, 227, 161)', 'rgba(0,0,0,0)']),  # couleur pour les bonnes réponses, transparent pour les mauvaises
        hoverinfo="skip",
        showlegend=False,
        textinfo='none'
    ))

    fig2.update_layout(
        title_text="Taux de bonnes réponses en général",
        annotations=[dict(text=f'{nb_bonnes_reponses}/{nb_questions}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig2)

