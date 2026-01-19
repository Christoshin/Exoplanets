import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

st.set_page_config(page_title="Visualization", page_icon="🪐", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/exoplanets_cleaned.csv")


df = load_data()

scales = px.colors.named_colorscales()
scales.sort()
selected_scale = st.sidebar.selectbox("Farbpalette wählen:", scales, index=78)
preview_colors = pc.sample_colorscale(selected_scale, 20)

gradient_css = ", ".join(preview_colors)
st.sidebar.markdown(f"**Vorschau:**")
st.sidebar.markdown(
    f'<div style="background: linear-gradient(to right, {gradient_css}); '
    f'height: 25px; width: 100%; border-radius: 5px; border: 1px solid #555;"></div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()
with st.sidebar.expander("Mögliche Farbpaletten", expanded=False):
    all_scales = sorted(px.colors.named_colorscales())

    for scale_name in all_scales:
        try:
            colors = px.colors.sample_colorscale(scale_name, 20)
            gradient = f"linear-gradient(to right, {', '.join(colors)})"

            st.markdown(
                f"""
                    <div style="
                        background: {gradient}; 
                        height: 25px; 
                        border-radius: 4px; 
                        margin-bottom: 10px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        color: white; 
                        font-weight: regular; 
                        font-size: 16px;
                        text-shadow: 
                            -0px -0px 5px #000,  
                             0px -0px 5px #000,
                            -0px  0px 5px #000,
                             0px  0px 5px #000;
                    ">
                        {scale_name}
                    </div>
                """,
                unsafe_allow_html=True,
            )
        except:
            continue

tab1, tab2, tab3 = st.tabs(
    [
        ":material/stacked_bar_chart: Histogramme",
        ":material/grid_on: Heatmap/Pairplot",
        ":material/grain: 3D-Pointcloud",
        # ":material/extension: Weitere Visualisierungen",
    ]
)

with tab1:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    col1, col2 = st.columns(2)
    histogram_type = col1.segmented_control(
        "Histogramtyp wählen",
        options=["Direkter Vergleich", "Form Vergleich"],
        default="Direkter Vergleich",
        selection_mode="single",
    )

    def reset_histogram():
        st.session_state.selectbox_1 = num_cols[0]
        st.session_state.selectbox_2 = None
        st.session_state.selectbox_3 = None
        st.session_state.selectbox_4 = None

    col2.button(
        ":material/restart_alt: Zurücksetzen",
        on_click=reset_histogram,
        help="Setzt die gesamte Auswahl zurück",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val1 = st.selectbox(
            "Wert 1 (Basis)", options=[None] + num_cols, index=1, key="selectbox_1"
        )
    with col2:
        val2 = st.selectbox(
            "Wert 2 (Vergleich)", options=[None] + num_cols, index=0, key="selectbox_2"
        )
    with col3:
        val3 = st.selectbox(
            "Wert 3 (Vergleich)", options=[None] + num_cols, index=0, key="selectbox_3"
        )
    with col4:
        val4 = st.selectbox(
            "Wert 4 (Vergleich)", options=[None] + num_cols, index=0, key="selectbox_4"
        )

    selected_metrics = [v for v in [val1, val2, val3, val4] if v is not None]

    if selected_metrics:
        df_melted = df[selected_metrics].melt(var_name="Metrik", value_name="Wert")

        if histogram_type == "Direkter Vergleich":
            fig = px.histogram(
                df_melted,
                x="Wert",
                color="Metrik",
                barmode="overlay",
                nbins=50,
                opacity=0.6,
                marginal="box",
                color_discrete_sequence=pc.sample_colorscale(selected_scale, 4),
            )

            fig.update_layout(
                title="Verteilung der ausgewählten Parameter",
                xaxis_title="Wert",
                yaxis_title="Anzahl (Frequenz)",
                legend_title="Parameter",
                bargap=0.05,
            )

            st.plotly_chart(fig, use_container_width=True)

        if histogram_type == "Form Vergleich":
            fig = px.histogram(
                df_melted,
                x="Wert",
                color="Metrik",
                facet_row="Metrik",
                nbins=100,
                marginal="box",
                color_discrete_sequence=pc.sample_colorscale(selected_scale, 4),
            )

            fig.update_xaxes(matches=None, showticklabels=True)
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.update_traces(bingroup=None)
            fig.update_layout(height=500, bargap=0.05)

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bitte wähle mindestens einen Wert in den Dropdowns aus.")

with tab2:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    def sync_axes():
        st.session_state.multiselect_axis_y = st.session_state.multiselect_axis_x

    def reset_axes():
        st.session_state.multiselect_axis_x = num_cols
        st.session_state.multiselect_axis_y = num_cols

    col1, col2 = st.columns([1, 1])
    with col1:
        plot_type = st.segmented_control(
            "Visualisierungstyp wählen",
            options=["Heatmap", "Pairplot"],
            default="Heatmap",
            selection_mode="single",
        )
    with col2:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.button(
                ":material/cached: Synchronisieren",
                on_click=sync_axes,
                help="Kopiert die Auswahl von Achse X nach Achse Y",
            )
        with c2:
            st.button(
                ":material/restart_alt: Zurücksetzen",
                on_click=reset_axes,
                help="Setzt die Auswahl von Achse X und Y zurück",
            )
    col1, col2 = st.columns([1, 1])
    with col1:
        x_selection = st.multiselect(
            f"{"Heatmap " if plot_type == "Heatmap" else "Pairplot "}Achse X",
            num_cols,
            default=num_cols,
            key="multiselect_axis_x",
        )
    with col2:
        y_selection = st.multiselect(
            f"{"Heatmap " if plot_type == "Heatmap" else "Pairplot "}Achse Y:",
            num_cols,
            default=num_cols,
            key="multiselect_axis_y",
        )

    if len(x_selection) >= 1 and len(y_selection) >= 1:
        if plot_type == "Heatmap":
            x_selection.sort()
            y_selection.sort()
            needed_cols = list(set(x_selection + y_selection))
            asym_corr = df[needed_cols].corr().loc[x_selection, y_selection]

            fig = px.imshow(
                asym_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=selected_scale,
                zmin=-1,
                zmax=1,
            )

            fig.update_coloraxes(cmid=0)
            st.plotly_chart(fig, width="stretch", height=1000)

        if plot_type == "Pairplot":

            def get_color_legend(n_colors, scale_name=selected_scale):
                try:
                    colors = pc.sample_colorscale(scale_name, n_colors)
                except:
                    colors = pc.sample_colorscale(selected_scale, n_colors)
                return colors

            color_options = [None] + df.columns.tolist()
            color_col = st.selectbox("Farb-kodierung nach:", color_options)
            if len(x_selection) <= 10 and len(y_selection) <= 10:
                x_selection.sort()
                y_selection.sort()

                def create_plotly_asym_pairplot(df, x_cols, y_cols, color_col=None):
                    fig = make_subplots(
                        rows=len(y_cols),
                        cols=len(x_cols),
                        shared_xaxes=True,
                        shared_yaxes=True,
                        column_titles=x_cols,
                        row_titles=y_cols,
                    )

                    marker_settings = dict(size=3, opacity=0.4)
                    legend_info = None

                    if color_col:
                        if (
                            df[color_col].dtype == "object"
                            or df[color_col].dtype.name == "category"
                        ):
                            cat_series = pd.Categorical(df[color_col])
                            color_values = cat_series.codes
                            marker_settings["colorscale"] = selected_scale
                            legend_info = dict(enumerate(cat_series.categories))
                        else:
                            color_values = df[color_col]
                            marker_settings["colorscale"] = selected_scale
                            marker_settings["showscale"] = True

                        marker_settings["color"] = color_values
                    else:
                        marker_settings["color"] = pc.sample_colorscale(
                            selected_scale, 3
                        )[1]

                    for row_idx, y_col in enumerate(y_cols):
                        for col_idx, x_col in enumerate(x_cols):
                            fig.add_trace(
                                go.Scattergl(
                                    x=df[x_col],
                                    y=df[y_col],
                                    mode="markers",
                                    marker=marker_settings,
                                    showlegend=False,
                                ),
                                row=row_idx + 1,
                                col=col_idx + 1,
                            )

                    fig.update_layout(height=250 * len(y_cols), width=250 * len(x_cols))
                    return fig, legend_info

                if x_selection and y_selection:
                    if plot_type == "Pairplot":
                        with st.spinner("Rendere Diagramm..."):
                            fig, legend_dict = create_plotly_asym_pairplot(
                                df, x_selection, y_selection, color_col
                            )
                            st.plotly_chart(fig, width="stretch")

                            if legend_dict:
                                with st.expander("Farblegende", expanded=True):
                                    colors = get_color_legend(
                                        len(legend_dict), scale_name=selected_scale
                                    )

                                    cols = st.columns(4)
                                    for i, (code, name) in enumerate(
                                        legend_dict.items()
                                    ):
                                        color = colors[i]
                                        legend_html = f'<span style="color:{color}; font-size: 20px;">●</span> **{name}**'
                                        cols[i % 4].markdown(
                                            legend_html, unsafe_allow_html=True
                                        )

            else:
                st.info(
                    "Bitte wähle aus Performance-Gründen maximal 10 Features pro Achse aus."
                )
    else:
        st.info("Bitte wähle für beide Achsen mindestens 1 Feature aus.")

with tab3:
    c1, c2, c3 = st.columns([1, 1, 1])

    filter_col = c1.selectbox(
        "Spalte zum Filtern", [None] + df.columns.tolist(), key="filter_col_sel"
    )

    df_filtered = df.copy()

    if filter_col:
        if df[filter_col].dtype == "object" or df[filter_col].nunique() < 20:
            options = sorted(df[filter_col].unique().tolist())
            selected_val = c2.multiselect(
                f"Werte für '{filter_col}' auswählen", options
            )
            if selected_val:
                df_filtered = df[df[filter_col].isin(selected_val)]

        else:
            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())

            range_val = c2.slider(
                f"Bereich für '{filter_col}' festlegen",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                format="%.2f",
            )
            df_filtered = df[
                (df[filter_col] >= range_val[0]) & (df[filter_col] <= range_val[1])
            ]

        st.info(f"Anzeigte Planeten: {len(df_filtered)} von {len(df)}")

    def get_color_legend(n_colors, scale_name=selected_scale):
        try:
            colors = pc.sample_colorscale(scale_name, n_colors)
        except:
            colors = pc.sample_colorscale(selected_scale, n_colors)
        return colors

    color_options = [None] + df.columns.tolist()
    color_col = c3.selectbox("Farb-kodierung nach:", color_options, key="Colorcoding2")

    legend_dict = None
    selected_scale = selected_scale

    if color_col:
        is_categorical = df[color_col].dtype == "object" or df[color_col].nunique() < 20

        if is_categorical:
            categories = df[color_col].astype("category").cat.categories
            df_filtered[f"{color_col}_code"] = pd.Categorical(
                df_filtered[color_col], categories=categories
            ).codes
            legend_dict = {i: name for i, name in enumerate(categories)}

            fig = px.scatter_3d(
                df_filtered,
                x="X_gal",
                y="Y_gal",
                z="Z_gal",
                color=f"{color_col}_code",
                opacity=0.6,
                color_continuous_scale=selected_scale,
                hover_name=color_col,
            )
            fig.update_coloraxes(showscale=False, cmin=0, cmax=len(categories) - 1)
        else:
            color_min = float(df[color_col].min())
            color_max = float(df[color_col].max())
            fig = px.scatter_3d(
                df_filtered,
                x="X_gal",
                y="Y_gal",
                z="Z_gal",
                color=color_col,
                opacity=0.6,
                color_continuous_scale=selected_scale,
                range_color=[color_min, color_max]
            )
    else:
        fig = px.scatter_3d(df_filtered, x="X_gal", y="Y_gal", z="Z_gal", opacity=0.6)
        fig.update_traces(marker=dict(color=pc.sample_colorscale(selected_scale, 3)[1]))

    fig.update_traces(marker=dict(size=3))

    fig.add_trace(
        go.Scatter3d(
            x=df["X_gal"],
            y=df["Y_gal"],
            z=df["Z_gal"],
            mode="markers",
            marker=dict(size=2, color="lightgrey", opacity=0.05),
            hoverinfo="skip",
            name="Gesamtbestand",
        )
    )

    x_range = [df["X_gal"].min(), df["X_gal"].max()]
    y_range = [df["Y_gal"].min(), df["Y_gal"].max()]
    z_range = [df["Z_gal"].min(), df["Z_gal"].max()]

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            annotations=[
                dict(
                    showarrow=True,
                    x=0,
                    y=0,
                    z=0,
                    text="Sonnensystem",
                    arrowcolor="white",
                    font=dict(color="white", size=12),
                    arrowsize=1,
                    arrowhead=2,
                    ax=50,
                    ay=-50,
                )
            ],
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key="map")

    if legend_dict:
        with st.expander("Farblegende", expanded=True):
            colors = get_color_legend(len(legend_dict), selected_scale)
            cols = st.columns(4)
            for i, (code, name) in enumerate(legend_dict.items()):
                color = colors[i]
                legend_html = (
                    f'<span style="color:{color}; font-size: 20px;">●</span> **{name}**'
                )
                cols[i % 4].markdown(legend_html, unsafe_allow_html=True)

# with tab4:
#     pass
