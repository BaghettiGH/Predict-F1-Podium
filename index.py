import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

st.set_page_config(
    page_title= "Predict F1 Podium",
    page_icon= "images/f1logo.png",
    layout='wide',
    menu_items={
        'GitHub':
    }



)










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

