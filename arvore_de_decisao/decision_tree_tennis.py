import pandas as pd
import streamlit as st
from sklearn import tree


st.title('Predição de jogos de tênis')

dfTenis = pd.DataFrame({
'Tempo'       : ['Chuvoso', 'Ensolarado', 'Ensolarado', 'Nublado', 'Chuvoso',
                 'Chuvoso', 'Nublado', 'Nublado', 'Ensolarado', 'Chuvoso',
                 'Nublado', 'Ensolarado',
                 'Ensolarado', 'Chuvoso'],
'Temperatura' : [22, 21, 27, 28, 21, 
                 18, 18, 22, 24, 20, 
                 27, 29, 22, 24],
'Umidade'     : [91,72,90,86,96,
                 70,65,90,70,80,
                 75,85,95,80],
'Vento'       : ['Sim','Não','Sim','Não','Não',
                 'Sim','Sim','Sim','Sim','Não',
                 'Não','Não','Não','Não'],
'Jogo'        : ['Não','Sim','Não','Sim','Sim',
                 'Não','Sim','Sim','Sim','Sim',
                 'Sim','Não','Não','Sim']
})

quali = ['Tempo', 'Vento']
quant = ['Temperatura', 'Umidade']

dfQualiDummies = pd.get_dummies(dfTenis[quali])
dfQuant = dfTenis[quant]

dfWork = pd.concat([dfQualiDummies, dfQuant ], axis=1 )
target = dfTenis['Jogo']

vento = st.radio('Tem vento?', ['Sim', 'Não'])
tempo = st.selectbox('Selecione o clima', ['Chuvoso', 'Ensolarado', 'Nublado'])
umidade = st.number_input('Informe a umidade', value=0)
temperatura = st.number_input('Informe a temperatura (°C)', value=0)

# inicia todos como 0 pra depois alterar
valores = [
    0, 0, 0, 0, 0, int(temperatura), int(umidade)
]

opcoes_vento = {
    'Não': valores[3],
    'Sim': valores[4]
}

opcoes_tempo = {
    'Chuvoso': valores[0],
    'Ensolarado': valores[1],
    'Nublado': valores[2]
}

opcoes_vento[vento] = 1
opcoes_tempo[tempo] = 1

arv = tree.DecisionTreeClassifier()
arv.fit(dfWork, target)
res = arv.predict([valores])

if res[0] == 'Sim':
    st.write("Terá jogo.")
else:
    st.write("Não terá jogo.")
