####################################################
# Importando as bibliotecas
####################################################

import streamlit as st
import pandas as pd
from numpy.core.numeric import True_
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

###################################################
@st.cache(persist= True)
###################################################

###################################################
### Importando os dados
###################################################

def load():
    """
    Busca o dataset e usa o label encoder para trata-lo
    """
    column_names = ['target',
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises?',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-root',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring',
    'veil-type',
    'veil-color',
    'ring-number',
    'ring-type',
    'spore-print-color',
    'population',
    'habitat']


    label = LabelEncoder()
    url = r'C:\Users\Pedro e Gustavo\Desktop\app\ML_Model\Data\agaricus-lepiota.data'
    df = pd.read_csv(url, header= None, names= column_names)
    for i in df.columns:
        df[i] = label.fit_transform(df[i])
    return df
df = load()
###################################################
# Checkbox - Dataset
###################################################

if st.sidebar.checkbox("Dataset", False):
    st.subheader("Veja o dataset o Dataset")
    st.write(df)

###################################################
# Train, test - split
###################################################

test_size = 0.3
RS = 0

@st.cache(persist=True)

def Split(df):
    """
    separação entre treino e tese do modelo preditivo
    Utilizando o x(dados analiticos) e y(dados alvo(target), na funcao train_test_split
    """
    y = df.target
    X = df.drop(columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=RS)

    # scaler = StandardScaler()
    # scaler.fit_transform(X_train)
    # scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = Split(df)

###################################################
# Vizualicao de metricas
###################################################

def plot_metrics(metrics_list):
    if "Confusion-Matrix" in metrics_list:
        st.subheader("Confusion-Matrix")
        plot_confusion_matrix(model, X_test, y_test, display_labels= class_names)
        st.pyplot()
    if "ROC-curve" in metrics_list:
        st.subheader("ROC-curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()
    if "Precision-recall-curve" in metrics_list:
        st.subheader("Precision-recall-curve")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()
class_names = ["Saudavel", "Venenoso"]

###################################################
### Criando as maquinas preditivas
###################################################

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.subheader("Escolha a Maquina Preditiva de Classificação")
clf = st.sidebar.selectbox("Classificador", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "KNN"))

###################################################
# Opção 1 - SVM
###################################################

# Construção da parte grafica e input
if clf == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparametros")
    C = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key='C')
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='Kernel')
    gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("Qual metrica utilizar?", ("Confusion-Matrix", "ROC-curve", "Precision-recall-curve"))
    test_size = st.sidebar.number_input("Test size", min_value=0.1, max_value=1.0, value=0.3, step=0.1, key='test_size')
    RS = st.sidebar.number_input("Random state", min_value=0, max_value=100, value=42, step=1, key='Random state')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Construção do SVM
    if st.sidebar.button("classify", key='classify'):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=np.unique(y_pred), average = 'weighted').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=np.unique(y_pred), average = 'micro').round(2))
        plot_metrics(metrics)

###################################################
# Opção 2 - Logistic Regression
###################################################

# Construção da parte grafica e input
if clf == "Logistic Regression":
    st.sidebar.subheader("Hyperparametros")
    C = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key='C_Lr')
    max_iter = st.sidebar.number_input("C (Regularization parameter)", min_value=100, max_value=500, step=1, key='max_iter')
    metrics = st.sidebar.multiselect("Qual metrica utilizar?", ("Confusion-Matrix", "ROC-curve", "Precision-recall-curve"))
    test_size = st.sidebar.number_input("Test size", min_value=0.1, max_value=1.0, value=0.3, step=0.1, key='test_size')
    RS = st.sidebar.number_input("Random state", min_value=0, max_value=100, value=42, step=1, key='Random state')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Construção do SVM
    if st.sidebar.button("classify", key='classify'):
        st.subheader("Logistic Regression results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=np.unique(y_pred), average = 'weighted').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=np.unique(y_pred), average = 'micro').round(2))
        plot_metrics(metrics)

###################################################
# Opção 3 - Random Forest
###################################################
# Construção da parte grafica e input
if clf == "Random Forest":
    st.sidebar.subheader("Hyperparametros")
    n_estimators= st
    metrics = st.sidebar.multiselect("Qual metrica utilizar?", ("Confusion-Matrix", "ROC-curve", "Precision-recall-curve"))
    test_size = st.sidebar.number_input("Test size", min_value=0.1, max_value=1.0, value=0.3, step=0.1, key='test_size')
    RS = st.sidebar.number_input("Random state", min_value=0, max_value=100, value=42, step=1, key='Random state')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Construção do SVM
    if st.sidebar.button("classify", key='classify'):
        st.subheader("Random Forest results")
        model = 
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=np.unique(y_pred), average = 'weighted').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=np.unique(y_pred), average = 'micro').round(2))
        plot_metrics(metrics)

###################################################
# Opção 4 - KNN
###################################################


