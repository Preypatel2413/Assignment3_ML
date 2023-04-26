import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

@st.cache_data
def make_data(dataset_option):
    opt = dataset_option.split()[0]
    if opt == "100":
        X, y = make_regression(n_samples=100,
                               n_features=10, n_informative=2,
                               random_state=2)
    elif opt == "200":
        X, y = make_regression(n_samples=200,
                               n_features=5, n_informative=2,
                               random_state=4)
    elif opt == "150":
        X, y = make_regression(n_samples=150,
                               n_features=7,n_informative=2,
                               random_state=2)
    else:
        X, y = make_regression(random_state=10)
    return X, y

def estimator_model(estimator_type):
    if estimator_type == "Linear regressor":
        model = LinearRegression()
    elif estimator_type == "Ridge regressor":
        model = Ridge()
    elif estimator_type == "Lasso regressor":
        model = Lasso()
    elif estimator_type == "SVR":
        model = SVR()
    else:
        model = LinearRegression()
    return model

options = ['100 samples with 10 features and 1 target', '200 samples with 5 features and 1 target', '150 samples with 7 features and 1 target']
dataset_option = st.selectbox('Select dataset size:', options)
X, y = make_data(dataset_option)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
fig = plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset")
plt.scatter(X[:,0], y)
st.pyplot(fig)

options = ['Linear regressor', 'Ridge regressor', 'Lasso regressor', 'SVR']
model_type = st.selectbox('Select model type to use:', options)
options = ['boosting', 'bagging', 'gradient descent']
ensemble_type = st.selectbox('Select the ensemble type:', options)
estimator_number = st.slider('n_estimators', 1, 20, 4)

fig = plt.figure()
if ensemble_type == "bagging":
    estimator_ = estimator_model(model_type)
    test_loss = []
    train_loss = []
    for i in range(1, estimator_number):
        model = BaggingRegressor( n_estimators=i, random_state=45)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        temp = mean_squared_error(y_test, y_pred)
        test_loss.append(temp)
        y_pred = model.predict(X_train)
        temp = mean_squared_error(y_train, y_pred)
        train_loss.append(temp)
    plt.plot(range(1, estimator_number), test_loss, label="test loss")
    plt.plot(range(1, estimator_number), train_loss, label="train loss")
elif ensemble_type == "gradient descent":
    test_loss = []
    estimator_ = estimator_model(model_type)
    for i in range(1, estimator_number):
        model = GradientBoostingRegressor( n_estimators=i, learning_rate=0.1, random_state=45)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_loss.append(mean_squared_error(y_test, y_pred))
    plt.plot(range(1, estimator_number), test_loss, label="test loss")
elif ensemble_type == "boosting":
    test_loss = []
    estimator_ = estimator_model(model_type)
    for i in range(1, estimator_number):
        model = AdaBoostRegressor(n_estimators=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_loss.append(mean_squared_error(y_test, y_pred))
    plt.plot(range(1, estimator_number), test_loss, label="test loss")

plt.legend()
plt.title("loss plot")
plt.xlabel("n_estimators")
plt.ylabel("loss")
st.pyplot(fig)

if st.button('Magic'):
    loss = []
    n_splits=5
    opts = ['Linear regressor', 'Ridge regressor', 'Lasso regressor', 'SVR']
    for opt in opts:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=32)
        cv_scores = []
        for train_index, val_index in kf.split(X_train):
            model = estimator_model(opt)
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
            model.fit(X_train_cv, y_train_cv)
            y_val_pred = model.predict(X_val_cv)
            cv_scores.append(mean_squared_error(y_val_cv, y_val_pred))
        loss.append(np.mean(cv_scores))
    best_model = estimator_model(opts[np.argmin(loss)])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    fig = plt.figure()
    plt.title(f"Best model fit is of {opts[np.argmin(loss)]}")
    plt.scatter(X_test[:,0], y_pred)
    plt.scatter(X_test[:,0], y_test)
    st.pyplot(fig)

    