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
                This is a streamlit application that predicts if a driver will finish on the top 3 by using their Starting positions and their Finishing positions in Free Practice 1,2, and 3.
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


def predictInstance(fp1Pos,fp2Pos,fp3Pos,startPos,driverFName,driverLName):
    driverName = driverFName +"-"+ driverLName
    if driverName in dfDt['driverId']:
        totalPod = dfDt.loc[dfDt['driverId']== driverName,'totalPodiums'].iloc[0]
        totalPole = dfDt.loc[dfDt['driverId']==driverName,'totalPolePositions'].iloc[0]
    else:
        st.warning(f"Driver Name is invalid")

    inputData = pd.DataFrame({
            'startingPos':[startPos],
            'fp1Pos':[fp1Pos],
            'fp2Pos':[fp2Pos],
            'fp3Pos':[fp3Pos],
            'totalPodiums': [totalPod],
            'totalPolePositions':[totalPole]
            })
    predict = decTreeFinal.predict(inputData)


def getDriverImg(driver):
    driver_images = {
        "albon": "images/albon.png",
        "alonso": "images/alonso.png",
        "bearman": "images/bearman.png",
        "gasly": "images/gasly.png",
        "hamilton": "images/hamilton.png",
        "hulkenberg": "images/hulkenberg.png",
        "kevin_magnussen": "images/kevin_magnussen.png",
        "lawson": "images/lawson.png",
        "leclerc": "images/leclerc.png",
        "max_verstappen": "images/max_verstappen.png",
        "norris": "images/norris.png",
        "ocon": "images/ocon.png",
        "perez": "images/perez.png",
        "piastri": "images/piastri.png",
        "russel": "images/russel.png",
        "sainz": "images/sainz.png",
        "stroll": "images/stroll.png",
        "tsunoda": "images/tsunoda.png",
    }
    

    



if st.session_state.page_selection == 'prediction':
    st.header('🏎️ F1 Podium Prediction ‍💨',)
    col = st.columns((3.5,4.5),gap ='medium')
    
    with col[0]:
        driver2024 = ()
        driverName = st.selectbox("Select Driver",)
        fp1Pos = st.number_input('Free Practice 1 Position', min_value = 1,max_value = 24, step=1)
        fp2Pos = st.number_input('Free Practice 2 Position', min_value = 1,max_value = 24, step=1)
        fp3Pos = st.number_input('Free Practice 3 Position', min_value = 1,max_value = 24, step=1)
        startPos = st.number_input('Starting Position', min_value = 1,max_value = 24, step=1)

        if st.button('Predict Result'):
            predictInstance(fp1Pos,fp2Pos,fp3Pos,startPos,driverFName,driverLName)
    with col[1]:
        st.image(image=driver_images,width=400)

elif st.session_state.page_selection == 'dataset':
    st.header('Dataset and Model')


