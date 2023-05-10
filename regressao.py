import pandas as pd
import csv
import sys
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def make_prediction(s1, s2, s3, s4):
    file_content = '''${fileContent}'''
    # Separar as linhas e os valores por tabulação (\t)
    lines = file_content.strip().split('\n')
    data_rows = [line.split('\t') for line in lines]

    # Criar um dicionário com as colunas e valores
    data_dict = {}
    header = data_rows[0]  # Primeira linha contém os nomes das colunas
    for col in header:
        data_dict[col] = []

    for row in data_rows[1:]:  # Linhas subsequentes contêm os valores
        for i, value in enumerate(row):
            col_name = header[i]
            data_dict[col_name].append(value)

    # Criar o DataFrame a partir do dicionário
    data = pd.DataFrame(data_dict)

    data = np.genfromtxt('diabetes.txt', delimiter='\t', skip_header=1)

    # Separar os dados em atributos (X) e rótulos (Y)
    X = data[:, :-1]
    Y = data[:, -1]

    # Converter as colunas numéricas para o tipo adequado (opcional)
    data[['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y']] = \
        data[['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y']].astype(float)

    # Identificar as variáveis relevantes para a progressão da diabetes
    X = data[['AGE', 'SEX', 'BMI', 'BP']]  # Variáveis independentes
    y = data['Y']  # Variável dependente

    # Separar as features (X) e o target (y)
    # X = data[['Age', 'Sex', 'BMI', 'BP']].values
    # y = data['Y'].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Criar o modelo de regressão linear
    model = LinearRegression()
    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict([s1, s2, s3, s4])

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Retornar o resultado
    (y_pred[0], mse, r2)