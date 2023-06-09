import streamlit as st
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title('Boosting in Regression')
st.write("Over here, we will try to visualise the effect of number of estimators in the ensembling methods for the regression and could tryout with different basic estimators")
st.write("Magic button will help you to find the best individual estimator on selected dataset.")
@st.cache_data
def make_data(dataset_option):
    opt = dataset_option.split()[0]
    if opt == "100":
        X, y = make_regression(n_samples=100,
                               n_features=10,n_informative=10,noise=50,
                               random_state=42)
    elif opt == "200":
        X, y = make_regression(n_samples=200,
                               n_features=5,n_informative=5,
                               random_state=56)
    elif opt == "150":
        X, y = make_regression(n_samples=150,
                               n_features=7,n_informative=2,
                               random_state=25)
    else:
        X, y = make_regression(random_state=10)
    return X, y

def estimator_model(estimator_type):
    if estimator_type == "Linear regressor":
        model = LinearRegression()
    elif estimator_type == "Decision Tree regressor":
        model = DecisionTreeRegressor()
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

if st.button('Magic'):
    loss = []
    n_splits=5
    opts = ['LinearRegressor', 'DecisionTreeRegressor', 'SVR']
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
    plt.scatter(X_test[:,0], y_pred, label = "Prediction Value")
    plt.scatter(X_test[:,0], y_test, label = "Real Vaule")
    plt.legend()
    st.pyplot(fig)

options = ['LinearRegressor', 'DecisionTreeRegressor', 'SVR']
model_type = st.selectbox('Select model type to use:', options)
options = ['boosting', 'bagging', 'gradient descent']
ensemble_type = st.selectbox('Select the ensemble type:', options)
estimator_number = st.slider('n_estimators', 1, 20, 4)

fig = plt.figure()
if ensemble_type == "bagging":
    estimator = estimator_model(model_type)
    test_loss = []
    train_loss = []
    for i in range(1, estimator_number):
        model = BaggingRegressor(base_estimator=estimator, n_estimators=i, random_state=45)
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
    estimator = estimator_model(model_type)
    for i in range(1, estimator_number):
        model = GradientBoostingRegressor( n_estimators=i, learning_rate=0.1, random_state=45)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_loss.append(mean_squared_error(y_test, y_pred))
    plt.plot(range(1, estimator_number), test_loss, label="test loss")
elif ensemble_type == "boosting":
    test_loss = []
    estimator = estimator_model(model_type)
    for i in range(1, estimator_number):
        model = AdaBoostRegressor(n_estimators=i, base_estimator=estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_loss.append(mean_squared_error(y_test, y_pred))
    plt.plot(range(1, estimator_number), test_loss, label="test loss")

plt.legend()
plt.title("loss plot")
plt.xlabel("n_estimators")
plt.ylabel("mean squared error loss")
st.pyplot(fig)
