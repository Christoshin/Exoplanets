import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Data Exploration", page_icon="🪐", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/exoplanets_cleaned.csv"), pd.read_csv(
        "data/exoplanets_uncleaned.csv", skiprows=296
    )


df, unclean_df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        ":material/search_insights: Übersicht",
        ":material/account_tree: Pipeline",
        ":material/stars_2: Datenqualität",
        ":material/text_compare: Vergleich",
    ]
)

with tab1:
    st.markdown(
        f"""
            ## Daten-Übersicht
            Mit dem im `Pipeline`-Tab beschriebenen Vorgehen wurde der originelle Datensatz bereinigt. 
            Aus den `{len(unclean_df)}` Zeilen und `{len(unclean_df.columns)}` Features wurden durch das Entfernen von Duplikaten und nicht nutzbaren Einträgen `{len(df)}` Zeilen und `{len(df.columns)}` Features extrahiert.
        """
    )
    st.divider()
    with st.expander("Unbereinigte Daten", expanded=True):
        col1, col2 = st.columns([3, 2])
        n_rows_cleaned = col1.slider(
            "Anzahl Zeilen bei den noch nicht bereinigten Daten", 5, 50, 10
        )
        col1.dataframe(unclean_df.head(n_rows_cleaned), width="stretch", height=220)
        col2.markdown("**Info:**")
        col2.markdown(f"Shape: {unclean_df.shape}")
        col2.dataframe(unclean_df.dtypes, width="stretch", height=220)

    with st.expander("Bereinigte Daten", expanded=True):
        col1, col2 = st.columns([2, 1])
        n_rows = col1.slider("Anzahl Zeilen bei den bereinigten Daten", 5, 50, 10)
        col1.dataframe(df.head(n_rows), width="stretch", height=220)
        col2.markdown("**Info:**")
        col2.markdown(f"Shape: {df.shape}")
        col2.dataframe(df.dtypes, width="stretch", height=220)

with tab2:
    st.markdown("## Die Schritte der data cleaning pipeline")
    st.divider()
    st.markdown(
        """
            ### Schritt 1: *Ersten Überblick verschaffen*
            In diesem Schritt habe ich den Datensatz betrachtet und mir einen Überblick verschaffen.
            Bei der originalen CSV bestehen die ersten Zeilen aus Beschreibungen der einzelnen Features, hier ein Ausschnitt davon: 
            ```
            ...
            # COLUMN dkin_flag:      Detected by Disk Kinematics
            # COLUMN soltype:        Solution Type
            # COLUMN pl_controv_flag: Controversial Flag
            # COLUMN pl_refname:     Planetary Parameter Reference
            # COLUMN pl_orbper:      Orbital Period [days]
            ...
            ```
            Basierend auf diesen Beschreibungen konnte ermittelt werden, welche Features wichtig sein könnten für die weitere Analyse und welche irrelevant sind.
            Zudem konnte man beim Betrachten des Datensatzes sehen, dass jeder Planet bei dem "default flag" nur einen Eintrag besitzt, wo dieser Flag 1 entspricht.
            Das bedeutet, dass es viele doppelte Einträge existieren, aber die meisten davon nicht der "default" ist und ignoriert werden kann. 
            
            ---
            ### Schritt 2: *Datensatz verkleinern 1/2*
            Hier wurden 3 wichtige Unterschritte durchgeführt um die Menge der Daten drastisch zu reduzieren. \n
            **Schritt 2.1:** die individuellen Exoplaneten extrahieren  
            Hier wurden nur die Einträge behalten, wo der default_flag 1 entspricht.\n
            **Schritt 2.2:** die unwichtigen Features entfernen  
            Basierend auf der Recherche aus Schritt 1 wurden hier 19 der 289 Features behalten.\n
            > Zeilen und Spalten übrig: 6052 x 19
            
            **Schritt 2.3:** die Exoplaneten mit zu wenig Daten filtern  
            Mit dem Ziel sowohl Planetentyp, als auch Habital Zone zu berechnen, 
            wurden alle Planeten entfernt, die für dieses Vorgehen nicht genug Informationen bereitstellen.
            > Zeilen und Spalten übrig: 1151 x 19
        """
    )
    with st.expander("Code Schritt 2", expanded=False):
        st.markdown(
            """
                ```python
                # SCHRITT 2.1
                data_reduced = data[data["default_flag"] == 1]
                # SCHRITT 2.2
                data_reduced = data_reduced[
                    [
                        "pl_name",  # Planeten name
                        "hostname",  # Name des Sterns des Planeten
                        "pl_rade",  # Radius des Planeten
                        "pl_masse",  # Masse des Planeten
                        "pl_msinie",  # Mindestmasse des Planets
                        "tran_flag",  # Transit Planet Flag
                        "pl_dens",  # Dichte des Planeten
                        "pl_orbsmax",  # Planeten Orbit Semi-Major Achse
                        "pl_orbper",  # Planeten Orbit Periode
                        "st_mass",  # Masse des Sterns
                        "st_teff",  # Stern-temperatur
                        "st_rad",  # Stern-radius
                        "st_lum",  # Luminosität des Sterns
                        "pl_insol",  # Planet Insolation
                        "pl_eqt",  # Planet Gleichgewichtstemperatur
                        "pl_orbeccen",  # Planet Orbit Exzentrizität
                        "sy_dist",
                        "glon",
                        "glat",
                    ]
                ]

                data_reduced
                # SCHRITT 2.3
                data_filtered_unfinished = data_reduced[
                    # Planetendichte, bzw. Typ bestimmen
                    (
                        (
                            data_reduced["pl_rade"].notna()
                            & (
                                data_reduced["pl_masse"].notna()
                                | (data_reduced["pl_msinie"].notna() & data_reduced["tran_flag"] == 1)
                            )
                        )
                        | data_reduced["pl_dens"].notna()
                    )
                    # Habitable Zone berechnung
                    & (
                        (
                            (
                                data_reduced["pl_orbsmax"].notna()
                                | (data_reduced["pl_orbper"].notna() & data_reduced["st_mass"].notna())
                            )
                            & (
                                data_reduced["st_teff"].notna()
                                & (data_reduced["st_rad"].notna() | data_reduced["st_lum"].notna())
                            )
                        )
                        | data_reduced["pl_insol"].notna()
                        | data_reduced["pl_eqt"].notna()
                    )
                    & data_reduced["pl_orbeccen"].notna()
                ]

                data_filtered_unfinished
                ```    
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 3: *Datenanalyse für weitere Reduzierung* 
            Bevor weiter gefiltert werden kann, muss ermittelt werden, ob man wirklich alle der Features braucht, und ob man alle der Zeilen gebrauchen kann.
            Eine erste Prüfung hat ergeben, dass es keine duplicates mehr gibt.
            Bisher wurden alle Features behalten, mit denen man potentiell die gewünschten Ergebnisse ermitteln kann, 
            allerdings kann es sein, dass nicht alle gebraucht werden.
            
        """
    )
    st.image("images/berechnung_planeten_klassifikation.png")
    st.markdown(
        """
            Hier in dem Bild ist zu sehen, dass es zwar möglich ist, dass die Planetenklassisfikation mittels Radius und minimal-masse durchgeführt werden kann,
            aber die anderen Werte ausreichend sind, sodass in diesem Fall die minimal-masse nicht weiter gebraucht wird. 
            Bei diesem Vorgehen wurde immer nach Genauigkeit des möglichen Ergebnisses priorisiert.
            Etwas Ähnliches ist bei st_lum und pl_orbsmax für die Berechnung der Habitable Zone zu sehen, wie hier in der Grafik gezeigt: 
        """
    )
    st.image("images/berechnung_planeten_HZ.png")
    with st.expander("Code Schritt 3", expanded=False):
        st.markdown(
            """
                ```python
                # Planet-Klassifikations Visualisierung
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                total_which_planet_type_calculation_data.plot.bar(
                    ax=axes[0], y="Count", legend=False, rot=0, color=chart_colors_three
                )

                axes[0].set_title("Mögliche Methode der Planeten-Klassifikation")
                axes[0].set_xlabel("Methode der Berechnung des Oberflächentyps")
                axes[0].set_ylabel("Anzahl der Exoplaneten")
                for container in axes[0].containers:
                    axes[0].bar_label(container, label_type="edge")

                which_planet_type_calculation_data.plot.bar(
                    ax=axes[1], y="Count", legend=False, rot=0, color=chart_colors_three
                )

                axes[1].set_title("Beste Methode der Planeten-Klassifikation")
                axes[1].set_xlabel("Methode der Berechnung des Oberflächentyps")
                axes[1].set_ylabel("Anzahl der Exoplaneten")
                for container in axes[1].containers:
                    axes[1].bar_label(container, label_type="edge")

                plt.tight_layout()
                plt.show()
                
                # HZ-Berechnungs Visualisierung
                total_with_teff_stlum_and_plorbsmax_count = data_filtered_unfinished[
                    data_filtered_unfinished["st_teff"].notna()
                    & data_filtered_unfinished["st_lum"].notna()
                    & data_filtered_unfinished["pl_orbsmax"].notna()
                ].shape[0]
                total_with_teff_strad_and_plorbsmax_count = data_filtered_unfinished[
                    data_filtered_unfinished["st_teff"].notna()
                    & data_filtered_unfinished["st_rad"].notna()
                    & data_filtered_unfinished["pl_orbsmax"].notna()
                ].shape[0]
                total_with_teff_stlum_and_plorbper_stmass_count = data_filtered_unfinished[
                    data_filtered_unfinished["st_teff"].notna()
                    & data_filtered_unfinished["st_lum"].notna()
                    & data_filtered_unfinished["pl_orbper"].notna()
                    & data_filtered_unfinished["st_mass"].notna()
                ].shape[0]
                total_with_teff_strad_and_plorbper_stmass_count = data_filtered_unfinished[
                    data_filtered_unfinished["st_teff"].notna()
                    & data_filtered_unfinished["st_rad"].notna()
                    & data_filtered_unfinished["pl_orbper"].notna()
                    & data_filtered_unfinished["st_mass"].notna()
                ].shape[0]
                total_with_plinsol = data_filtered_unfinished[
                    data_filtered_unfinished["pl_insol"].notna()
                ].shape[0]
                total_with_stlum_and_plorbsmax = data_filtered_unfinished[
                    data_filtered_unfinished["st_lum"].notna()
                    & data_filtered_unfinished["pl_orbsmax"].notna()
                ].shape[0]
                total_with_pleqt = total_with_plinsol = data_filtered_unfinished[
                    data_filtered_unfinished["pl_eqt"].notna()
                ].shape[0]

                total_in_HZ_calculation_data = pd.DataFrame(
                    {
                        "Method-name": [
                            "HZ-borders (st_teff & st_lum & pl_orbsmax)",
                            "HZ-borders (st_teff & st_rad & pl_orbsmax)",
                            "HZ-borders (st_teff & st_lum & pl_orbper & st_mass)",
                            "HZ-borders (st_teff & st_rad & pl_orbper & st_mass)",
                            "Insolation (pl_insol)",
                            "Insolation (st_lum & pl_orbsmax)",
                            "Equilibrium Temperature (pl_eqt)",
                        ],
                        "Count": [
                            total_with_teff_stlum_and_plorbsmax_count,
                            total_with_teff_strad_and_plorbsmax_count,
                            total_with_teff_stlum_and_plorbper_stmass_count,
                            total_with_teff_strad_and_plorbper_stmass_count,
                            total_with_plinsol,
                            total_with_stlum_and_plorbsmax,
                            total_with_pleqt,
                        ],
                    }
                ).set_index("Method-name")


                with_teff_stlum_and_plorbsmax_count = data_filtered_unfinished[
                    data_filtered_unfinished["st_teff"].notna()
                    & data_filtered_unfinished["st_lum"].notna()
                    & data_filtered_unfinished["pl_orbsmax"].notna()
                ].shape[0]
                without_teff_stlum_and_plorbsmax_count = data_filtered_unfinished[
                    ~(
                        data_filtered_unfinished["st_teff"].notna()
                        & data_filtered_unfinished["st_lum"].notna()
                        & data_filtered_unfinished["pl_orbsmax"].notna()
                    )
                ]
                with_teff_strad_and_plorbsmax_count = without_teff_stlum_and_plorbsmax_count[
                    without_teff_stlum_and_plorbsmax_count["st_teff"].notna()
                    & without_teff_stlum_and_plorbsmax_count["st_rad"].notna()
                    & without_teff_stlum_and_plorbsmax_count["pl_orbsmax"].notna()
                ].shape[0]
                without_teff_strad_and_plorbsmax_count = without_teff_stlum_and_plorbsmax_count[
                    ~(
                        without_teff_stlum_and_plorbsmax_count["st_teff"].notna()
                        & without_teff_stlum_and_plorbsmax_count["st_rad"].notna()
                        & without_teff_stlum_and_plorbsmax_count["pl_orbsmax"].notna()
                    )
                ]
                with_teff_stlum_and_plorbper_stmass_count = without_teff_strad_and_plorbsmax_count[
                    without_teff_strad_and_plorbsmax_count["st_teff"].notna()
                    & without_teff_strad_and_plorbsmax_count["st_lum"].notna()
                    & without_teff_strad_and_plorbsmax_count["pl_orbper"].notna()
                    & without_teff_strad_and_plorbsmax_count["st_mass"].notna()
                ].shape[0]
                without_teff_stlum_and_plorbper_stmass_count = without_teff_strad_and_plorbsmax_count[
                    ~(
                        without_teff_strad_and_plorbsmax_count["st_teff"].notna()
                        & without_teff_strad_and_plorbsmax_count["st_lum"].notna()
                        & without_teff_strad_and_plorbsmax_count["pl_orbper"].notna()
                        & without_teff_strad_and_plorbsmax_count["st_mass"].notna()
                    )
                ]
                with_teff_strad_and_plorbper_stmass_count = (
                    without_teff_stlum_and_plorbper_stmass_count[
                        without_teff_stlum_and_plorbper_stmass_count["st_teff"].notna()
                        & without_teff_stlum_and_plorbper_stmass_count["st_rad"].notna()
                        & without_teff_stlum_and_plorbper_stmass_count["pl_orbper"].notna()
                        & without_teff_stlum_and_plorbper_stmass_count["st_mass"].notna()
                    ].shape[0]
                )
                without_teff_strad_and_plorbper_stmass_count = (
                    without_teff_stlum_and_plorbper_stmass_count[
                        ~(
                            without_teff_stlum_and_plorbper_stmass_count["st_teff"].notna()
                            & without_teff_stlum_and_plorbper_stmass_count["st_rad"].notna()
                            & without_teff_stlum_and_plorbper_stmass_count["pl_orbper"].notna()
                            & without_teff_stlum_and_plorbper_stmass_count["st_mass"].notna()
                        )
                    ]
                )

                with_plinsol = without_teff_strad_and_plorbper_stmass_count[
                    without_teff_strad_and_plorbper_stmass_count["pl_insol"].notna()
                ].shape[0]
                without_plinsol = without_teff_strad_and_plorbper_stmass_count[
                    ~without_teff_strad_and_plorbper_stmass_count["pl_insol"].notna()
                ]
                with_stlum_and_plorbsmax = without_plinsol[
                    without_plinsol["st_lum"].notna() & without_plinsol["pl_orbsmax"].notna()
                ].shape[0]
                without_stlum_and_plorbsmax = without_plinsol[
                    ~(without_plinsol["st_lum"].notna() & without_plinsol["pl_orbsmax"].notna())
                ]
                with_pleqt = without_stlum_and_plorbsmax[
                    without_stlum_and_plorbsmax["pl_eqt"].notna()
                ].shape[0]


                best_in_HZ_calculation_data = pd.DataFrame(
                    {
                        "Method-name": [
                            "HZ-borders (st_teff & st_lum & pl_orbsmax)",
                            "HZ-borders (st_teff & st_rad & pl_orbsmax)",
                            "HZ-borders (st_teff & st_lum & pl_orbper & st_mass)",
                            "HZ-borders (st_teff & st_rad & pl_orbper & st_mass)",
                            "Insolation (pl_insol)",
                            "Insolation (st_lum & pl_orbsmax)",
                            "Equilibrium Temperature (pl_eqt)",
                        ],
                        "Count": [
                            with_teff_stlum_and_plorbsmax_count,
                            with_teff_strad_and_plorbsmax_count,
                            with_teff_stlum_and_plorbper_stmass_count,
                            with_teff_strad_and_plorbper_stmass_count,
                            with_plinsol,
                            with_stlum_and_plorbsmax,
                            with_pleqt,
                        ],
                    }
                ).set_index("Method-name")

                fig, axes = plt.subplots(2, 1, figsize=(10, 10))
                total_in_HZ_calculation_data.plot.barh(
                    ax=axes[0], y="Count", legend=False, rot=0, color=chart_colors_seven
                )

                best_in_HZ_calculation_data.plot.barh(
                    ax=axes[1], y="Count", legend=False, rot=0, color=chart_colors_seven
                )

                axes[0].set_title("Mögliche Methode der Habitability Zone (HZ) Prüfung")
                axes[0].set_xlabel("Anzahl der Exoplaneten")
                axes[0].set_ylabel("Methode der Prüfung der HZ")
                for container in axes[0].containers:
                    axes[0].bar_label(container, label_type="edge")
                axes[0].invert_yaxis()

                axes[1].set_title("Beste Methode der Habitability Zone (HZ) Prüfung")
                axes[1].set_xlabel("Anzahl der Exoplaneten")
                axes[1].set_ylabel("Methode der Prüfung der HZ")
                for container in axes[1].containers:
                    axes[1].bar_label(container, label_type="edge")
                axes[1].invert_yaxis()
                ```
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 4: *Datensatz verkleinern 2/2*
            Basierend auf den Erkenntnissen aus Schritt 3 und mit weiterer Recherche wurde festgelegt, welche Zeilen besser weggelassen werden, 
            da bei denen die Berechnung und Klassifikation entweder zu inakkurat oder sogar nicht möglich wäre. Es wurden folgende Zeilen entfernt:
            - Zeilen ohne wo st_teff, st_rad, pl_orbper und st_mass nicht definiert sind
            - Zeilen wo pl_masse, pl_rade und sy_dist nicht definiert sind
            
            Außerdem konnten weitere Features entfernt werden, die nun nicht mehr für die Berechnung gebraucht werden. Folgende Features wurden entfernt:
            - tran_flag
            - pl_msinie
            - pl_insol
            - pl_eqt
            > Zeilen und Spalten übrig: 1118 x 15
            
            Nach dieser Filterung ist genau ein Planet aufgefallen, bei dem die Werte für "pl_orbper" und "st_mass" fehlen, und zwar bei dem Planeten "HD 135344 A b". 
            Diese wurden manuell recherchiert und hinzugefügt:  
            pl_orbper = 44 Jahre [(Quelle)](https://science.nasa.gov/exoplanet-catalog/hd-135344-a-b/)  
            st_mass = 2.32 Sonnenmassen [(Quelle)](https://en.wikipedia.org/wiki/HD_135344)
        """
    )
    with st.expander("Code Schritt 4", expanded=False):
        st.markdown(
            """
                ```python
                # Entfernen von Planeten, wo die berechnung ungenau oder unmöglich ist:
                data_removed_rows = data_filtered_unfinished.drop(
                    without_teff_strad_and_plorbper_stmass_count.index
                )
                data_removed_rows = data_removed_rows[
                    data_removed_rows["pl_masse"].notna()
                    & data_removed_rows["pl_rade"].notna()
                    & data_removed_rows["sy_dist"].notna()
                ]

                # Entfernen von Features, die nicht mehr gebraucht werden:
                columns_to_drop = [
                    "tran_flag",
                    "pl_msinie",
                    "pl_insol",
                    "pl_eqt",
                ]
                data_filtered = data_removed_rows.drop(columns=columns_to_drop)

                # Manuelles hinzufügen der Werte von ""
                data_filtered.loc[2234, "pl_orbper"] = 44.0 * days_in_year
                data_filtered.loc[2234, "st_mass"] = 2.32
                ```    
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 5: *Datentyp-konvertierung*
            Hier wurde mit folgendem Code geprüft, ob es Einträge gibt, die nicht dem richtigen Datentyp entsprechen:
            ```python
            display(data_filtered.dtypes)
            display(data_filtered["pl_name"].apply(type).value_counts())
            display(data_filtered["hostname"].apply(type).value_counts())
            ```
            Das Ergebnis zeigte, dass keine Datentyp-konvertierung durchgeführt werden muss.
            
            ---
            ### Schritt 6: *Standardisierung und Konsistenz*
            Zur Sicherheit wurde mit den folgenden Code-Zeilen sichergestellt, dass die Strings alle standardisiert sind:
            ```
            data_filtered["pl_name"] = data_filtered["pl_name"].str.strip()
            data_filtered["hostname"] = data_filtered["hostname"].str.strip()
            ```
            
            ---
            ### Schritt 7: *NaN-Daten Bereinigung*
            In der folgenden Missingno-matrix ist zu sehen, dass es dennoch eine Menge unvollständige Einträge gab, zu viele um diese einfach zu entfernen:
        """
    )
    st.image("images/missingno_matrix_pre_nan.png")
    st.markdown(
        """
            Die fehlenden Werte bei pl_dens, st_lum und pl_orbsmax wurden aus anderen vorhandenen Werten berechnet und eingefügt. Folgende Mengen wurden hinzugefügt:
            - 183 Einträge bei pl_dens
            - 495 Einträge bei st_lum
            - 160 Einträge bei pl_orbsmax
            
            Daraufhin sind alle NaN-Werte bereinigt worden wie man in dieser missingno-matrix sehen kann:
            
        """
    )
    st.image("images/missingno_matrix_post_nan.png")
    with st.expander("Code Schritt 7", expanded=False):
        st.markdown(
            """
                ```python
                # pl_dens
                pre_dens_calc_nan_count = data_filtered[data_filtered["pl_dens"].isna()].shape[0]

                RHO_EARTH = 5.514
                data_filtered.loc[data_filtered["pl_dens"].isna(), "pl_dens_calc"] = (
                    data_filtered["pl_masse"] / (data_filtered["pl_rade"] ** 3) * RHO_EARTH
                )

                data_filtered["pl_dens"] = data_filtered["pl_dens"].fillna(
                    data_filtered["pl_dens_calc"]
                )

                data_filtered.drop("pl_dens_calc", axis=1, inplace=True)

                post_dens_calc_calculated_lines = (
                    pre_dens_calc_nan_count - data_filtered[data_filtered["pl_dens"].isna()].shape[0]
                )
                print(
                    f"Density: {post_dens_calc_calculated_lines} of {pre_dens_calc_nan_count} missing values filled"
                )

                # st_lum
                pre_stlum_calc_nan_count = data_filtered[data_filtered["st_lum"].isna()].shape[0]

                T_SUN = 5780
                data_filtered.loc[data_filtered["st_lum"].isna(), "st_lum_calc"] = np.log10(
                    data_filtered["st_rad"] ** 2 * (data_filtered["st_teff"] / T_SUN) ** 4
                )

                data_filtered["st_lum"] = data_filtered["st_lum"].fillna(data_filtered["st_lum_calc"])

                data_filtered.drop("st_lum_calc", axis=1, inplace=True)

                post_stlum_calc_calculated_lines = (
                    pre_stlum_calc_nan_count - data_filtered[data_filtered["st_lum"].isna()].shape[0]
                )
                print(
                    f"Star Luminosity: {post_stlum_calc_calculated_lines} of {pre_stlum_calc_nan_count} missing values filled"
                )

                # pl_orbsmax
                pre_orbsmax_calc_nan_count = data_filtered[data_filtered["pl_orbsmax"].isna()].shape[0]

                data_filtered.loc[data_filtered["pl_orbsmax"].isna(), "pl_orbsmax_calc"] = (
                    (data_filtered["pl_orbper"] / days_in_year) ** 2 * data_filtered["st_mass"]
                ) ** (1 / 3)

                data_filtered["pl_orbsmax"] = data_filtered["pl_orbsmax"].fillna(
                    data_filtered["pl_orbsmax_calc"]
                )

                data_filtered.drop("pl_orbsmax_calc", axis=1, inplace=True)

                post_orbsmax_calc_calculated_lines = (
                    pre_orbsmax_calc_nan_count
                    - data_filtered[data_filtered["pl_orbsmax"].isna()].shape[0]
                )
                print(
                    f"Planet Max Orbit: {post_orbsmax_calc_calculated_lines} of {pre_orbsmax_calc_nan_count} missing values filled"
                )
                ```
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 8: *Outlier Bereinigung*
            In folgenden Box-plots sieht man direkt eine Menge Einträge, die wie Outlier aussehen:
        """
    )
    st.image("images/outlier_pre.png")
    st.markdown(
        """
            Allerdings sind viele dieser hohen Werte möglich. 2 Filter konnten dennoch gesetzt werden:
            - Planetenradien > 25 sind physikalisch unmöglich
            - Planetenmassen > 4200 sind keine Planeten mehr, sondern braune Zwergsterne
            Diese wurden aus dem Datensatz entfernt. Es gab 18 Planeten mit einer Dichte über 25 und 9 Planeten mit einer Masse über 4200, insgesamt wurden 24 Outlier entfernt.
            > Zeilen und Spalten übrig: 1094 x 15
            
            Durch diese Änderungen sind die Boxplots recht gleich geblieben, allerdings gab es kleine Änderungen:
        """
    )
    st.image("images/outlier_post.png")
    with st.expander("Code Schritt 8", expanded=False):
        st.markdown(
            """
                ```python
                def showOutlier():
                    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
                    flier_style = dict(
                        marker="o",
                        markersize=4,
                        color="darkred",
                        markeredgecolor="black",
                        linestyle="none",
                    )

                    for i, col_name in enumerate(
                        [
                            "pl_rade",
                            "pl_masse",
                            "pl_dens",
                            "pl_orbsmax",
                            "pl_orbper",
                            "st_mass",
                            "st_teff",
                            "st_rad",
                            "st_lum",
                            "pl_orbeccen",
                        ]
                    ):
                        row = i // 5
                        col_idx = i % 5
                        ax = axes[row, col_idx]

                        ax.boxplot(data_filtered[col_name].dropna(), whis=0.1, flierprops=flier_style)
                        ax.set_title(col_name)
                        ax.set_xlabel(
                            f"min: {data_filtered[col_name].min()} max: {data_filtered[col_name].max()}"
                        )
                        ax.set_xticks([])

                showOutlier()
                plt.show()

                MAX_RHO = 25.0
                MAX_MASS = 4200.0

                outlier_count = {}
                outlier_count["pl_dens > 25.0"] = (data_filtered["pl_dens"] > MAX_RHO).sum()
                outlier_count["pl_masse > 4200"] = (data_filtered["pl_masse"] > MAX_MASS).sum()

                for criterion, count in outlier_count.items():
                    if count > 0:
                        print(f"  {criterion: <20}: {count} Planeten")

                mask_outliers_combined = (data_filtered["pl_dens"] > MAX_RHO) | (
                    data_filtered["pl_masse"] > MAX_MASS
                )

                num_removed_total = mask_outliers_combined.sum()
                data_filtered = data_filtered[~mask_outliers_combined].copy()

                print(f"Planet Outliers removed: {num_removed_total}")

                showOutlier()
                plt.show()
                ```
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 9: *Feature Engineering*
            In diesem Schritt wurden folgende Features engineered: 
            - Planetentyp [(Quelle)](https://ethz.ch/content/dam/ethz/special-interest/phys/particle-physics/quanz-group-dam/documents-old-s-and-p/Courses/ExtrasolarPlanetsFS2016/exop2016_chapter3_part1_UPDATED.pdf)
            - HZ-Grenzen und Distanzen [(Quelle)](https://iopscience.iop.org/article/10.1088/0004-637X/765/2/131)
            - XYZ-Koordinaten des Planetensystems in Heliocentrischen Koordinaten
            
            Hier sind die dazu passenden Grafiken:
        """
    )
    st.image("images/feature_engineering_planettyp.png")
    st.image("images/feature_engineering_HZ.png")
    fig = px.scatter_3d(
        df,
        x="X_gal",
        y="Y_gal",
        z="Z_gal",
        color="pl_rade",
        opacity=0.6,
        color_continuous_scale="Plasma",
    )
    fig.update_traces(marker=dict(size=5))

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
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        """
        > Zeilen und Spalten übrig: 1094 x 26
        """
    )
    with st.expander("Code Schritt 9", expanded=False):
        st.markdown(
            """
                ```python
                # Planeten-typ bestimmen
                cond_ter = (data_filtered["pl_dens"] > 3.0) & (data_filtered["pl_masse"] <= 1.0)
                cond_sea = (
                    (data_filtered["pl_dens"] > 3.0)
                    & (data_filtered["pl_masse"] > 1.0)
                    & (data_filtered["pl_masse"] <= 10.0)
                )
                cond_mea = (data_filtered["pl_dens"] > 3.0) & (data_filtered["pl_masse"] > 10.0)
                cond_ice = (data_filtered["pl_dens"] > 2.0) & (data_filtered["pl_dens"] <= 3.0)
                cond_mne = (
                    (data_filtered["pl_dens"] > 1.3)
                    & (data_filtered["pl_dens"] <= 2.0)
                    & (data_filtered["pl_masse"] <= 10.0)
                )
                cond_lne = (
                    (data_filtered["pl_dens"] > 1.3)
                    & (data_filtered["pl_dens"] <= 2.0)
                    & (data_filtered["pl_masse"] > 10.0)
                )
                cond_gas = data_filtered["pl_dens"] <= 1.3

                data_filtered["planet_type"] = np.select(
                    [cond_ter, cond_sea, cond_mea, cond_ice, cond_mne, cond_lne, cond_gas],
                    [
                        "Terrestrial",
                        "Super-Earth",
                        "Mega-Earth",
                        "Icy-Solid",
                        "Mini-Neptune",
                        "Neptune-like",
                        "Gas-giant",
                    ],
                    default="Uncertain",
                )
                
                print(
                    f"planets without values for mass, radius and density: {data_filtered[data_filtered["pl_masse"].isna() | data_filtered["pl_dens"].isna() | data_filtered["pl_rade"].isna()].shape[0]}"
                )
                print(
                    f"NaN or Null Planet Types: {data_filtered[data_filtered["planet_type"].isna() | data_filtered["planet_type"].isnull()].shape[0]}"
                )
                print(
                    f'Planets with "Uncertain" Planet Type: {data_filtered[data_filtered["planet_type"] == "Uncertain"].shape[0]}'
                )
                display(data_filtered.head())

                count_by_type = data_filtered["planet_type"].value_counts()
                sizes_types = count_by_type.values
                labels_types = count_by_type.index
                total_planets = data_filtered.shape[0]

                plt.pie(
                    count_by_type,
                    labels=None,
                    autopct=None,
                    startangle=90,
                    colors=chart_colors_seven,
                )

                custom_labels_planettypes = []
                for i, sizes_types in enumerate(sizes_types):
                    percent_types = (sizes_types / total_planets) * 100
                    new_label = f"{labels_types[i]} ({sizes_types} Planeten | {percent_types:.1f}%)"
                    custom_labels_planettypes.append(new_label)

                plt.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    labels=custom_labels_planettypes,
                    title="Anzahl Planeten pro Typ",
                )

                plt.title(f"Verteilung von {total_planets} Planeten nach Planettyp")
                plt.show()
                
                
                # in HZ oder nicht
                dT = data_filtered["st_teff"] - T_SUN

                data_filtered["lum_ratio"] = 10 ** data_filtered["st_lum"]

                # Recent Venus, RV (Conservative Inner)
                S_RV = 1.7753
                a_RV = 1.4316e-4
                b_RV = 2.9875e-9
                c_RV = -7.5702e-12
                d_RV = -1.1635e-15
                S_eff_RV = S_RV + a_RV * dT + b_RV * (dT**2) + c_RV * (dT**3) + d_RV * (dT**4)

                # Early Mars, EM (Conservative Outer)
                S_EM = 0.3179
                a_EM = 5.4513e-5
                b_EM = 1.5313e-9
                c_EM = -2.7786e-12
                d_EM = -4.8997e-16
                S_eff_EM = S_EM + a_EM * dT + b_EM * (dT**2) + c_EM * (dT**3) + d_EM * (dT**4)

                # Runaway Greenhouse, RG (Optimistic Inner)
                S_RG = 1.0512
                a_RG = 1.3242e-4
                b_RG = 1.5418e-8
                c_RG = -7.9895e-12
                d_RG = -1.8328e-15
                S_eff_RG = S_RG + a_RG * dT + b_RG * (dT**2) + c_RG * (dT**3) + d_RG * (dT**4)

                # Maximum Greenhouse, Max_CO2 (Optimistic Outer)
                S_MC = 0.3438
                a_MC = 5.8942e-5
                b_MC = 1.6558e-9
                c_MC = -3.0045e-12
                d_MC = -5.2983e-16
                S_eff_MC = S_MC + a_MC * dT + b_MC * (dT**2) + c_MC * (dT**3) + d_MC * (dT**4)

                data_filtered["hz_start_con"] = np.sqrt(data_filtered["lum_ratio"] / S_eff_RV)
                data_filtered["hz_end_con"] = np.sqrt(data_filtered["lum_ratio"] / S_eff_EM)
                data_filtered["hz_start_opt"] = np.sqrt(data_filtered["lum_ratio"] / S_eff_RG)
                data_filtered["hz_end_opt"] = np.sqrt(data_filtered["lum_ratio"] / S_eff_MC)

                data_filtered.drop("lum_ratio", axis=1, inplace=True)

                conditions_hz = [
                    # Too Hot
                    (data_filtered["pl_orbsmax"] < data_filtered["hz_start_opt"]),
                    # Optimistic in HZ
                    (
                        (data_filtered["pl_orbsmax"] >= data_filtered["hz_start_opt"])
                        & (data_filtered["pl_orbsmax"] < data_filtered["hz_start_con"])
                    )
                    | (
                        (data_filtered["pl_orbsmax"] <= data_filtered["hz_end_opt"])
                        & (data_filtered["pl_orbsmax"] > data_filtered["hz_end_con"])
                    ),
                    # Conservative in HZ
                    (data_filtered["pl_orbsmax"] >= data_filtered["hz_start_con"])
                    & (data_filtered["pl_orbsmax"] <= data_filtered["hz_end_con"]),
                    # Too Cold
                    (data_filtered["pl_orbsmax"] > data_filtered["hz_end_opt"]),
                ]

                choices_hz = ["Too_hot", "Optimistic_HZ", "Conservative_HZ", "Too_cold"]
                data_filtered["in_hz"] = np.select(conditions_hz, choices_hz, default="Undefined")

                count_by_HZ = data_filtered["in_hz"].value_counts()
                sizes_HZ = count_by_HZ.values
                labels_HZ = count_by_HZ.index
                total_planets = data_filtered.shape[0]

                plt.pie(
                    count_by_HZ,
                    labels=None,
                    autopct=None,
                    startangle=90,
                    colors=chart_colors_four,
                )

                custom_labels_planetHZ = []
                for i, sizes_HZ in enumerate(sizes_HZ):
                    percent_types = (sizes_HZ / total_planets) * 100
                    new_label = f"{labels_HZ[i]} ({sizes_HZ} Planeten | {percent_types:.1f}%)"
                    custom_labels_planetHZ.append(new_label)

                plt.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    labels=custom_labels_planetHZ,
                    title="Anzahl Planeten pro HZ-Bereich",
                )

                plt.title(f"Verteilung von {total_planets} Planeten nach HZ")
                plt.show()
                
                # Distanz zu HZ

                dist_to_inner = data_filtered["hz_start_opt"] - data_filtered["pl_orbsmax"]
                dist_to_outer = data_filtered["pl_orbsmax"] - data_filtered["hz_end_opt"]

                # Erstellung der finalen Metrik: Die kleinste (am wenigsten extreme) Distanz
                conditions = [
                    dist_to_inner > 0,
                    dist_to_outer > 0,
                    (dist_to_inner <= 0) & (dist_to_outer <= 0)
                ]

                choices = [
                    dist_to_inner, 
                    dist_to_outer, 
                    0 
                ]

                data_filtered["hz_dist"] = np.select(conditions, choices, default=np.nan)
                
                
                # Planetensystem xyz-koordinaten

                glon_rad = np.deg2rad(data_filtered["glon"])
                glat_rad = np.deg2rad(data_filtered["glat"])
                r = data_filtered["sy_dist"]

                data_filtered["X_gal"] = r * np.cos(glat_rad) * np.cos(glon_rad)

                data_filtered["Y_gal"] = r * np.cos(glat_rad) * np.sin(glon_rad)

                data_filtered["Z_gal"] = r * np.sin(glat_rad)
                ```
            """
        )
    st.markdown(
        """
            ---
            ### Schritt 10: *Validierung*
            
            Da zuerst der Plan war zu untersuchen, welche Planeten bewohnbar wären und welche nicht, konnte nun basierend auf den ermittelten Werten geschaut werden, welche Planeten die Kriterien erfüllen.
            Letztendlich hat nur 1 Planet alle Kriterien erfüllt:
        """
    )
    hz_condition = df["in_hz"].isin(["Conservative_HZ", "Optimistic_HZ"])
    type_condition = df["planet_type"].isin(["Terrestrial", "Super-Earth"])
    eccentricity_condition = df["pl_orbeccen"] < 0.5

    top_candidates = df[hz_condition & type_condition & eccentricity_condition].copy()

    st.dataframe(top_candidates)
    st.markdown(
        """
            Im vergleich zum Original (39119 Zeilen, 289 Spalten) hat der bereinigte Datensatz (1094 Zeilen, 26 Spalten) eine Retention von 2.8%. 
        """
    )
    st.markdown(
        """
            ---
            ### Schritt 11: *Speichern*
            Der Bereinigte Datensatz wurde anschließend in eine neue CSV-Datei names `exoplanets_cleaned.csv` geschrieben.
        """
    )
    with st.expander("Code Schritt 11", expanded=False):
        st.markdown(
            """
                ```python
                output = 'exoplanets_cleaned.csv'
                data_filtered.to_csv(output, index=False)
                print(f"Gespeichert: {output}")
                print(f"Zeilen: {data_filtered.shape[0]}")
                print(f"Spalten: {data_filtered.shape[1]}")
                ```
            """
        )

with tab3:
    st.markdown(
        f"""
            ## Die Qualität der Daten
            Hier wird die Daten-qualität des bereinigten Datensatzes beschrieben.
        """
    )
    st.dataframe(df.head())
    st.caption("Head des Datensatzes")
    st.divider()
    with st.expander("Herkunft und Verlässlichkeit", expanded=True):
        st.markdown(
            """
                Bei dem Datensatz handelt es sich um das [Exoplaneten Archiv von Nasa](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS).  
                
                Der Datensatz zeigt bei weitem nicht die gesamte Menge der Exoplaneten. Neben dem Aspekt, dass die meisten noch nicht gefunden wurden,
                führt NASA schon eine gewisse Filterung durch um die Datenqualität zu garantieren.  
                Außerdem wurden bei der Bereinigung eine Menge Einträge entfernt, die zu wenig Daten hatten, Ausreißer waren oder sonstige Mängel aufwiesen.
                
                Bei einigen Einträgen ist eine Abweichung angegeben. Im Datensatz sind die Werte gegeben als 3 Features, die obere Unsicherheit, die unterer Unsicherheit und ein zentraler Nennwert (Best fit).
                Für die Untersuchungen in diesem Projekt wurden die Nennwerte verwendet.
            """
        )
    with st.expander("Selection Bias", expanded=True):
        st.markdown(
            """
                Da es nur [bestimmte Methoden](https://sci.esa.int/web/exoplanets/-/60655-detection-methods) gibt zur Entdeckung von Exoplaneten, sind vermehrt Exoplaneten im Datensatz vorhanden, die leichter durch diese Methoden gefunden werden.
                Das beinhaltet Planeten die folgende Merkmale aufweisen:
                - Hohe Planeten-masse
                - Großer Planeten-radius
                - Planet nah am Stern
                - Planeten mit kleinem, kühlem Stern
                
                Bei der Analyse der Ergebnisse kann man außerdem Erkennen, dass Planeten mit einem kleinen Radius eher näher am Sonnensystem liegen, als die mit großem Radius.
                Das kann man in folgender Visualisierung gut erkennen:
            """
        )
        fig_2 = px.scatter_3d(
            df,
            x="X_gal",
            y="Y_gal",
            z="Z_gal",
            color="pl_rade",
            opacity=0.61,
            color_continuous_scale="Plasma",
        )
        fig_2.update_traces(marker=dict(size=5))

        fig_2.update_layout(
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
        fig_2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_2, width="stretch")
        st.write(
            "Nah an unserem Sonnensystem ist eine größere Menge an Planeten mit kleinem Radius (Dunkel-lila) zu erkennen, während Planeten weiter weg (Orange-gelb) eher größer sind."
        )
    with st.expander("NaN-Werte und Platzhalter", expanded=True):
        st.write(
            "Der Datensatz weist nach der Bereinigung keine fehlenden Werte oder Platzhalter auf. Das ist unter anderem in der missingno-matrix zu sehen."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("NaN-Einträge", df.isna().any(axis=1).sum())
        col2.metric("null-Einträge", df.isnull().any(axis=1).sum())
        col3.metric(
            "Platzhalter-Einträge",
            np.isinf(df.select_dtypes(include=np.number)).values.sum()
            + (df == "").sum().sum()
            + (df == "undefined").sum().sum(),
        )
        fig = msno.matrix(df).get_figure()
        st.pyplot(fig)
    with st.expander("Duplikate", expanded=True):
        st.write(
            "Nach der Bereinigung sind keine Duplikate mehr im Datensatz vorhanden. Jeder Planetenname ist einzigartig."
        )
        col1, col2 = st.columns(2)
        col1.metric("Doppelte Zeilen", len(df[df.duplicated()]))
        col2.metric("Doppelte Planetennamen", len(df[df.duplicated("pl_name")]))
    with st.expander("Outlier", expanded=True):
        st.write(
            "In den folgenden Boxplots ist zu erkennen, dass es eine Menge Exoplaneten mit vergleichsweise hohen oder niedrigen Werten gibt:"
        )
        st.image("images/outlier_post.png")
        st.write(
            "Diese Werte sind allerdings realistisch, da die Werte der Exoplaneten extrem verschieden sind. Somit wurden diese bei der Auswertung mit beachtet."
        )
        st.write(
            "Die Einträge, die die physisch möglichen Grenzen überschritten haben, wurden entfernt."
        )
    with st.expander("Standardisierung", expanded=True):
        st.markdown(
            """
                Da die Daten bereits ziemlich einheitlich bereitgestellt wurden, mussten nur wenige Standardisierungsschritte durchgeführt werden.
                Das Ergebnis ist ein Datensatz, wo alle Zahlenwerte den Datentyp float64 haben und die Namens- und Kategorialischen Spalten den Typ von string (object) aufweisen.
                
                #### Datentyp
                Hier ist die Anzahl der Spalten pro Datentyp:
            """
        )
        dtypes_df = df.dtypes.value_counts().reset_index()
        dtypes_df.columns = ["Datentyp", "Anzahl der Spalten"]
        st.table(dtypes_df)
        st.markdown(
            """
                Die 4 Spalten des Typs object sind wie erwartet pl_name, hostname und die kategorialen Spalten planet_type und in_hz. Der Rest hat den typ float64.
                
                #### Einheitlichkeit
                Auch bei den Zahlenwerten wurde darauf geachtet, dass diese das korrekte Format und die korrekte Einheit haben. Die Koordinaten der Planeten z.B. sind Heliocentrische Koordinaten, was bedeutet, dass das Sonnensystem im Mittelpunkt steht.  
                Jeder String wurde zur Sicherheit von Whitespace vor oder hinter dem String befreit und die kategorialen Features sind einheitlich erstellt worden.
            """
        )

with tab4:
    st.markdown("## Ein direkter Vergleich der Daten vor und nach Daten-bereinigung")
    st.divider()
    col1, seperator_col, col2 = st.columns([5, 0.1, 5])
    with seperator_col:
        st.markdown(
            """
                <div style="
                    border-left: 1px solid #ddd;
                    height: 2350px;
                    margin: auto;
                    width: 1px;
                "></div>
            """,
            unsafe_allow_html=True,
        )
    with col1:
        st.subheader("Ursprünglicher Datensatz")
        st.download_button(
            label=":material/download: Download CSV ",
            data=unclean_df.to_csv(index=False),
            file_name="raw_data.csv",
            mime="text/csv",
        )
        st.dataframe(unclean_df.head())
        st.divider()
        st.subheader("Shape")
        sub_col1, sub_col2 = st.columns([1, 1])
        sub_col1.metric("Zeilen", len(unclean_df))
        sub_col2.metric("Features", len(unclean_df.columns))
        st.divider()
        st.subheader("NaN-Werte (missingno)")
        fig = msno.matrix(unclean_df, labels=False).get_figure()
        st.pyplot(fig)
        st.divider()
        st.subheader("Duplikate")
        sub_col1, sub_col2, sub_col3 = st.columns([1,1,1])
        sub_col1.metric("Doppelte Zeilen", len(unclean_df[unclean_df.duplicated()]))
        sub_col2.metric("Doppelte Planeten", len(unclean_df[unclean_df.duplicated("pl_name")]))
        sub_col3.metric("Doppelte Sterne", len(unclean_df[unclean_df.duplicated("hostname")]))
        st.divider()
        st.subheader("Outlier")
        sub_col1, sub_col2 = st.columns([1,1])
        fig = px.violin(unclean_df, y="pl_rade", points=False, box=True, title="Radius")
        sub_col1.plotly_chart(fig, width='stretch')
        fig = px.violin(unclean_df, y="pl_masse", points=False, box=True, title="Masse")
        sub_col2.plotly_chart(fig, width='stretch')
        fig = px.violin(unclean_df, y="st_mass", points=False, box=True, title="Sternmasse")
        sub_col1.plotly_chart(fig, width='stretch')
        fig = px.violin(unclean_df, y="sy_dist", points=False, box=True, title="Distanz zum Sonnensystem")
        sub_col2.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Bereinigter Datensatz")
        st.download_button(
            label=":material/download: Download CSV",
            data=df.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        st.dataframe(df.head())
        st.divider()
        st.subheader("Shape")
        sub_col1, sub_col2 = st.columns([1, 1])
        sub_col1.metric("Zeilen", len(df))
        sub_col2.metric("Features", len(df.columns))
        st.divider()
        st.subheader("NaN-Werte (missingno)")
        fig = msno.matrix(df, labels=False).get_figure()
        st.pyplot(fig)
        st.divider()
        st.subheader("Duplikate")
        sub_col1, sub_col2, sub_col3 = st.columns([1,1,1])
        sub_col1.metric("Doppelte Zeilen", len(df[df.duplicated()]))
        sub_col2.metric("Doppelte Planeten", len(df[df.duplicated("pl_name")]))
        sub_col3.metric("Doppelte Sterne", len(df[df.duplicated("hostname")]))
        st.divider()
        st.subheader("Outlier")
        sub_col1, sub_col2 = st.columns([1,1])
        fig = px.violin(df, y="pl_rade", points=False, box=True, title="Radius")
        sub_col1.plotly_chart(fig, width='stretch')
        fig = px.violin(df, y="pl_masse", points=False, box=True, title="Masse")
        sub_col2.plotly_chart(fig, width='stretch')
        fig = px.violin(df, y="st_mass", points=False, box=True, title="Sternmasse")
        sub_col1.plotly_chart(fig, width='stretch')
        fig = px.violin(df, y="sy_dist", points=False, box=True, title="Distanz zum Sonnensystem")
        sub_col2.plotly_chart(fig, width='stretch')

