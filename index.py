import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

st.set_page_config(
    page_title= "F1 Podium Prediction",
    page_icon= "images/f1logo.png",
    layout='wide'
)
alt.themes.enable('dark')

with st.sidebar:
    st.image('images/f1logo.png',use_column_width='auto')
    st.title('F1 Podium Prediction')
    st.markdown("""
                This is a streamlit application that predicts if a driver will finish on the top 3 by using their: 

                - Starting positions 
                - Finishing positions in Free Practice 1,2,3.
                - Driver's total podiums
                - Driver's total pole positions
                """)


    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = 'prediction'
    def set_page_selection(page):
        st.session_state.page_selection = page

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):
        st.session_state.page_selection = 'prediction'
    
    if st.button("Dataset and Model", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'










#--------Reading Dataset---------
dfStartingPos = pd.read_csv("assets/f1db-races-starting-grid-positions.csv")
dfRaceResults = pd.read_csv("assets/f1db-races-race-results.csv")
dfDriver = pd.read_csv("assets/f1db-drivers.csv")
dfFp1 = pd.read_csv("assets/f1db-races-free-practice-1-results.csv")
dfFp2 = pd.read_csv("assets/f1db-races-free-practice-2-results.csv")
dfFp3 = pd.read_csv("assets/f1db-races-free-practice-3-results.csv")
dfRaces = pd.read_csv("assets/f1db-races.csv")

#--------Initializing Dataset---------
col1 = ['raceId','positionDisplayOrder','driverId']
dfStartingPos = dfStartingPos[col1]
dfRaceResults = dfRaceResults[col1]
dfFp1 = dfFp1[col1]
dfFp2 = dfFp2[col1]
dfFp3 = dfFp3[col1]
dfDriver = dfDriver[['id','name','totalPodiums','totalPolePositions']]
dfRaces = dfRaces[['id','year','grandPrixId']]

dfStartingPos.rename(columns={"positionDisplayOrder":'startingPos'},inplace=True)
dfRaceResults.rename(columns={"positionDisplayOrder":'finishPos'},inplace=True)
dfFp1.rename(columns={"positionDisplayOrder":"fp1Pos"},inplace=True)
dfFp2.rename(columns={"positionDisplayOrder":"fp2Pos"},inplace=True)
dfFp3.rename(columns={"positionDisplayOrder":"fp3Pos"},inplace=True)
dfDriver.rename(columns={"id":'driverId'},inplace=True)
dfRaces.rename(columns={"id":"raceId"},inplace=True)

mergedRace = pd.merge(dfRaceResults,dfStartingPos,on=['raceId','driverId'],how='left')
mergedFp1 = pd.merge(mergedRace,dfFp1,on=['raceId','driverId'],how='left')
mergedFp2 = pd.merge(mergedFp1, dfFp2,on=['raceId','driverId'],how='left')
mergedFp3 = pd.merge(mergedFp2,dfFp3,on=['raceId','driverId'],how='left')

mergedDriver = pd.merge(mergedFp3,dfDriver, on=['driverId'],how='inner')
df = pd.merge(mergedDriver,dfRaces,on=['raceId'],how='inner')
df.dropna(inplace=True)
df['podiumFinish'] = df['finishPos'].apply(lambda x: 1 if x<=3 else 0)

rearrangeCol = ['driverId','name','totalPodiums','totalPolePositions','raceId','grandPrixId','startingPos','fp1Pos','fp2Pos','fp3Pos','podiumFinish','finishPos']
df = df[rearrangeCol]

#---Machine Learning-----

dfDt = df
features = ['startingPos','fp1Pos','fp2Pos','fp3Pos','totalPodiums','totalPolePositions']
XDt = dfDt[features]
yDt = dfDt['podiumFinish']
X_train, X_test, y_train, y_test = train_test_split(XDt,yDt,test_size=0.2,random_state=1)
decTreeFinal = DecisionTreeClassifier(max_leaf_nodes=22, random_state=1)
decTreeFinal.fit(X_train,y_train)


def predictInstance(fp1Pos,fp2Pos,fp3Pos,startPos,driverId):

    totalPod = dfDt.loc[dfDt['driverId']== driverId,'totalPodiums'].iloc[0]
    totalPole = dfDt.loc[dfDt['driverId']==driverId,'totalPolePositions'].iloc[0]
    inputData = pd.DataFrame({
            'startingPos':[startPos],
            'fp1Pos':[fp1Pos],
            'fp2Pos':[fp2Pos],
            'fp3Pos':[fp3Pos],
            'totalPodiums': [totalPod],
            'totalPolePositions':[totalPole]
            })
    predict = decTreeFinal.predict(inputData)
    return predict


def setDriverImg(driver):
    driverImg = {
    "alexander-albon": "images/albon.png",
    "fernando-alonso": "images/alonso.png",
    "valterri-bottas": "images/bottas.png",
    "franco-colapinto": "images/colapinto.png",
    "pierre-gasly": "images/gasly.png",
    "lewis-hamilton": "images/hamilton.png",
    "nico-hulkenberg": "images/hulkenberg.png",
    "kevin-magnussen": "images/kevin_magnussen.png",
    "liam-lawson": "images/lawson.png",
    "charles-leclerc": "images/leclerc.png",
    "max-verstappen": "images/max_verstappen.png",
    "lando-norris": "images/norris.png",
    "sergio-perez": "images/perez.png",
    "oscar-piastri": "images/piastri.png",
    "george-russell": "images/russel.png",
    "carlos-sainz-jr": "images/sainz.png",
    "lance-stroll": "images/stroll.png",
    "yuki-tsunoda": "images/tsunoda.png",
    "guanyu-zhou": "images/zhou.png"
    }

    st.image(driverImg[driver],width=400)

    
prediction = None


if st.session_state.page_selection == 'prediction':
    st.header('🏎️ F1 Prediction 2024 ‍💨',)
    col = st.columns((3.5,4.5),gap ='medium')
    col1= st.columns((3.5,4.5),gap = 'medium')    
    with col[0]:
        driver2024 = {
            'Alexander Albon': 'alexander-albon',
            'Fernando Alonso':'fernando-alonso',
            'Valterri Bottas':'valterri-bottas',
            'Franco Colapinto': 'franco-colapinto',
            'Pierre Gasly': 'pierre-gasly',
            'Lewis Hamilton':'lewis-hamilton',
            'Nico Hulkenberg': 'nico-hulkenberg',
            'Kevin Magnussen': 'kevin-magnussen',
            'Liam Lawson': 'liam-lawson',
            'Charles Leclerc':'charles-leclerc',
            'Max Verstappen': 'max-verstappen',
            'Lando Norris':'lando-norris',
            'Sergio Perez':'sergio-perez',
            'Oscar Piastri':'oscar-piastri',
            'George Russell':'george-russell',
            'Carlos Sainz': 'carlos-sainz-jr',
            'Lance Stroll':'lance-stroll',
            'Yuki Tsunoda':'yuki-tsunoda',
            'Zhou Guanyu':'guanyu-zhou'         
        }   


        driverName = st.selectbox("Select Driver",driver2024.keys())
        driverId = driver2024[driverName]
        fp1Pos = st.number_input('Free Practice 1 Position', min_value = 1,max_value = 20, step=1)
        fp2Pos = st.number_input('Free Practice 2 Position', min_value = 1,max_value = 20, step=1)
        fp3Pos = st.number_input('Free Practice 3 Position', min_value = 1,max_value = 20, step=1)
        startPos = st.number_input('Starting Position', min_value = 1,max_value = 20, step=1)
    
        
        if st.button('Predict Result'):
            prediction = predictInstance(fp1Pos,fp2Pos,fp3Pos,startPos,driverId)

    with col[1]:
        setDriverImg(driverId)
        if prediction is not None:
            if prediction[0]==1:
                st.write(f" **{driverName}** has a high chance of finishing on the podium!")
            else:
                st.write(f" **{driverName}** has a low chance of finishing on the podium.")
        
        
        

elif st.session_state.page_selection == 'dataset':
    st.header('Dataset and Model')
    st.subheader('Dataset')
    st.dataframe(df)
    st.markdown("""
                Special thanks to F1DB for the comprehensive F1 Database! Checkout [F1DB](https://github.com/f1db/f1db) for the full database

                """)
    
    st.subheader('Machine Learning Model')
    st.markdown("""

                This streamlit application uses **Decision Tree Classifier** as the machine learning model to predict if the driver will finish on the podium.

                Features used for training are:

                - Free Practice 1 Position
                - Free Practice 2 Position
                - Free Practice 3 Position
                - Starting Position
                - Driver's Total Podium Count
                - Driver's Total Pole Position Count

                The model has an accuracy of **91%**.
                
                """)


