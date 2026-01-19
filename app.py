import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Exoplaneten", page_icon="🪐", layout="wide")
st.title("Exoplaneten")
st.markdown(
    """ Diese App analysiert [dein Thema] und erstellt 
Vorhersagen.
👈 **Wähle eine Seite im Sidebar!** """
)


@st.cache_data
def load_data():
    return pd.read_csv("data/exoplanets_cleaned.csv")


df = load_data()
col1, col2, col_seperator, col3, col4, col5 = st.columns([1, 0.5, 0.5, 1, 1, 1])
col1.metric("Zeilen", len(df))
col2.metric("Features", len(df.columns))
col_seperator.markdown(
    """
        <div style="border-left: 1px solid #ccc; height: 100px; margin-left: 50%; margin-bottom: 50%;"></div>
    """,
    unsafe_allow_html=True,
)
col3.metric(
    "Gesteinswelten",
    len(df[df["planet_type"].isin(["Terrestrial", "Super-Earth", "Mega-Earth"])]),
)
col4.metric(
    "Neptun-ähnliche",
    len(df[df["planet_type"].isin(["Neptune-like", "Mini-Neptune", "Icy-Solid"])]),
)
col5.metric("Gasriesen", len(df[df["planet_type"].isin(["Gas-giant"])]))
st.dataframe(df.head())

st.header("3D-Karte der Exoplaneten")
st.markdown(
    "Eine interaktive 3D-Karte der Exoplaneten in Heliocentrische Koordinaten, farblich kodiert nach dem Radius der Planeten."
)

fig = px.scatter_3d(
    df,
    x="X_gal",
    y="Y_gal",
    z="Z_gal",
    color="pl_rade",
    opacity=0.6,
    color_continuous_scale="Sunsetdark",
)
fig.update_traces(marker=dict(size=2))

fig.update_layout(
    scene=dict(
        annotations=[
            dict(
                showarrow=True,
                x=0,
                y=0,
                z=0,
                text="unser Sonnensystem",
                # bgcolor="white",
                opacity=0.8,
                arrowcolor="white",
                arrowsize=1,
                arrowhead=2,
            )
        ]
    )
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig, width='stretch')
