####################################################
# Importando as bibliotecas
####################################################


import streamlit as st
import pandas as pd
from numpy.core.numeric import True_
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

###################################################
### Importando os dados
###################################################

def load():
    """
    Busca o dataset e usa o label encoder para trata-lo
    """
    label = LabelEncoder()
    url = r'C:\Users\Pedro e Gustavo\Desktop\app\ML_Model\Data\supermarket_sales - Sheet1.csv'
    df = pd.read_csv(url)
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
    y = df['Total']
    X = df.drop(columns=["Total", "Tax 5%"])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=RS)

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
class_names = ["Total", "Unit price"]

###################################################
### Criando as maquinas preditivas
###################################################

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.subheader("Escolha a Maquina Preditiva de Classificação")
clf = st.sidebar.selectbox("Classificador", ("Support Vector Machine (SVM)", "Linear Regression", "Random Forest", "KNN"))

###################################################
# Opção 1 - SVM
###################################################
if clf == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparametros")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='Kernel')
    gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("Qual metrica utilizar?", ("Confusion-Matrix", "ROC-curve", "Precision-recall-curve"))
    test_size = st.sidebar.number_input("Test size", 0.1, 1.0, step=0.1, key='test_size')
    RS = st.sidebar.number_input("Random state", 0, 100, step=1, key='Random state')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.button("classify", key='classify'):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("precision: ", precision_score(y_test, y_pred, labels=class_names, average='weighted')).round(2)
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names)).round(2)
        plot_metrics(metrics)

###################################################
# Opção 2 - Linear Regression
###################################################
###################################################
# Opção 3 - Random Forest
###################################################
###################################################
# Opção 4 - KNN
###################################################


