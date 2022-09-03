import streamlit as st
import pandas as pd

rioAptos = pd.read_csv('https://raw.githubusercontent.com/mvinoba/notebooks-for-binder/master/dados.csv')
# rioAptos

bairro = st.selectbox('Escolha o bairro', rioAptos['bairro'].unique())

aptos = rioAptos[rioAptos['bairro'] == bairro]
minimo = int(aptos['preco'].min())
maximo = int(aptos['preco'].max())

faixa = st.slider("Selecione a faixa de preço", minimo, maximo, value=(minimo, maximo))

st.write(f"Preço minímo: {faixa[0]}")
st.write(f"Preço máximo: {faixa[1]}")

idx = aptos['preco'].between(faixa[0], faixa[1], inclusive=True)
st.write(f"Número de apartamentos encontrados em {bairro} nessa faixa de preço: {aptos[idx].shape[0]}")

st.subheader("Apartamentos dentro dessa faixa de preço")
aptos[idx]
