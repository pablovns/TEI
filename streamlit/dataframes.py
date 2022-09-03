import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import geobr

#download dos geodataframes
gdf_estados = geobr.read_state()
gdf_municipios = geobr.read_municipality()

st.title("Tópicos especiais em informática")
st.caption('Cidades')

#escolha do estado
estados = gdf_estados.sort_values('abbrev_state')['abbrev_state']
estado = st.selectbox('Estado', estados, index=17)
st.write(f'Estado escolhido: {estado}')

#escolha do município
municipios = gdf_municipios[gdf_municipios['abbrev_state'] == estado]['name_muni']
municipio = st.selectbox('Município', municipios)
st.write(f'Município escolhido: {municipio}')
st.write(f'Quantidade de municípios em {estado}: {len(municipios)}')
# st.write(gdf_municipios.head(2))

# atividade: plotar o mapa dos municipios e estados escolhidos