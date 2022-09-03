import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import geobr


st.title("Tópicos Especiais em Informática")
st.caption("Atividade envolvendo Streamlit e GeoPandas")

gdf_muni = geobr.read_municipality()
gdf_estados = geobr.read_state()

gdf_muni = gdf_muni.sort_values(by="name_muni", ascending=True)
municipios = gdf_muni["name_muni"]

# escolha dos municípios
municipios_escolha = st.multiselect("Escolha os municípios: ", municipios)

# plotar
fig, ax = plt.subplots()

gdf_municipios_escolha = gdf_muni[gdf_muni["name_muni"].isin(municipios_escolha)]
gdf_municipios_escolha.plot(ax=ax, color="k", alpha=1)

gdf_estados_escolhidos = gdf_estados[gdf_estados["abbrev_state"].isin(gdf_municipios_escolha["abbrev_state"])]
gdf_estados_escolhidos.plot(ax=ax, color="g", alpha=0.4)

st.subheader("Mapa dos munícipios e seus estados: ")

# apresentar a figura
st.pyplot(fig)
