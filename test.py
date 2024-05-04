import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

def init():
    # Load data
    dTrain = pd.read_csv("./event-recommendation-engine-challenge/train.csv")
    dTest = pd.read_csv("./event-recommendation-engine-challenge/test.csv")
    publicSol = pd.read_csv("./event-recommendation-engine-challenge/public_leaderboard_solution.csv")

    publicIdx = dTest[dTest['user'].isin(publicSol['User'])].index
    publicIdxUnique = pd.Index(dTest['user'].unique()).intersection(publicSol['User'])

    numTrain = len(dTrain) + len(publicIdx)
    numAll = len(dTrain) + len(dTest)

    # Combine data
    d1 = pd.concat([dTrain.drop(columns=['column5', 'column6']), 
                    dTest.iloc[publicIdx], 
                    dTest.drop(publicIdx)])

    allUsers = d1['user']
    allEvents = d1['event']
    uniqueUsers = allUsers.unique()
    numUser = len(uniqueUsers)
    numUserTrain = len(d1.iloc[:numTrain]['user'].unique())

    # Indexing users
    allUsersIdx = pd.factorize(allUsers)[0] + 1

    userIdx = {k: np.flatnonzero(allUsersIdx == k) for k in np.unique(allUsersIdx)}
    user2event = {k: allEvents[allUsersIdx == k].tolist() for k in np.unique(allUsersIdx)}

    # Prepare target vector for model
    targetPublic = np.zeros(len(publicSol))
    a, b = len(np.unique(dTrain['user'])) + 1, len(np.unique(dTrain['user'])) + len(publicSol) - 1
    for i, (user, event) in enumerate(zip(user2event[a:b+1], publicSol['Events'])):
        targetPublic[i] = (user == event).astype(int).sum()
    target = np.concatenate([dTrain['interested'], targetPublic])

    # Convert timestamps to structured formats
    d1['timestamp'] = pd.to_datetime(d1['timestamp'])
    d1['month'] = d1['timestamp'].dt.month
    d1['date'] = d1['timestamp'].dt.day
    d1['hour'] = d1['timestamp'].dt.hour
    d1['week'] = d1['timestamp'].dt.weekday

    return d1, target, allUsersIdx, uniqueUsers, userIdx, user2event, numUserTrain

# Factorize top k frequent values
def FactorTopkFrequentVals(series, k=20):
    counts = series.value_counts().nlargest(k)
    return pd.Series(np.where(series.isin(counts.index), series, "others"), index=series.index)

# Normalize data
def l2_normalize(a):
    s2 = np.sum(a**2)
    return a / np.sqrt(s2) if s2 > 0 else a

# Prediction model
def makePrediction(X_train, y_train, X_test, users_test, vars_idx1, vars_idx2, events_test):
    model1 = GradientBoostingRegressor()
    model1.fit(X_train[vars_idx1], y_train)
    pred1 = model1.predict(X_test[vars_idx1])

    ratio1 = 0.05
    y2 = y_train - ratio1 * np.exp(np.exp(pred1))
    model2 = GradientBoostingRegressor()
    model2.fit(X_train[vars_idx2], y2)
    pred2 = model2.predict(X_test[vars_idx2])

    pred = pred2 + 0.32 * pred1
    return pred

d1, target, allUsersIdx, uniqueUsers, userIdx, user2event, numUserTrain = init()

# Assuming some columns are in your data for demonstration
vars_idx1 = ['column1', 'column2']
vars_idx2 = ['column3', 'column4']

# This will be defined according to how you want to split your data
X_train, y_train, X_test, users_test = d1, target, d1, allUsersIdx

pred = makePrediction(X_train, y_train, X_test, users_test, vars_idx1, vars_idx2, user2event)
print(pred)

# To save predictions or any other output
np.savetxt("predictions.csv", pred, delimiter=",")
