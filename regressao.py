def makePrediction(s1, s2, s3, s4):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Carregar a base de dados do diabetes a partir do link
    url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
    data = pd.read_csv(url, sep='\\t')

    # Identificar as variáveis relevantes para a progressão da diabetes
    X = data[['BMI', 'BP', 'S4', 'S5']]  # Variáveis independentes
    y = data['Y']  # Variável dependente

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar um modelo de regressão linear
    model = LinearRegression()

    # Pré-processamento dos dados de entrada
    regr_quad = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = regr_quad.fit_transform(X_train)
    X_test_poly = regr_quad.transform(X_test)

    # Treinar o modelo usando o conjunto de treinamento
    model.fit(X_train_poly, y_train)

    # Fazer previsões usando o conjunto de teste
    y_pred = model.predict(X_test_poly)

    # Avaliar o desempenho do modelo

    # RMSE (Raiz quadrada do erro médio quadrático)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # R^2 (Coeficiente de determinação)
    r2 = r2_score(y_test, y_pred)

    # Fazer a previsão da progressão da diabetes usando o modelo treinado
    input_data = [[s1, s2, s3, s4]]
    input_data_poly = regr_quad.transform(input_data)
    predicted_progression = model.predict(input_data_poly)[0]

    return rmse, r2, predicted_progression