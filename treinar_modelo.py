"""
Passo 4 - Programe o Mínimo
Treinamento essencial de um modelo de Classificação (Regressão Logística)
utilizando a base de dados completa com divisão entre treino e teste.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Carregamento dos dados
df = pd.read_csv('dataset_estudantes_real.csv')

# 2. Definição das variáveis de entrada (X) e saída esperada (y)
features = ['horas_sono', 'minutos_rede_social', 'horas_estudo_planejado', 'horas_atividade_fisica']
X = df[features]
y = df['procrastinou']

# 3. Divisão em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 4. Treinamento do Modelo Mínimo
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train, y_train)

# 5. Avaliação do Modelo no conjunto de teste
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("=" * 60)
print("     PASSO 4: MODELO TREINADO E AVALIADO COM SUCESSO")
print("=" * 60)
print(f"Total de registros na base: {len(df)}")
print(f"Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}")
print(f"\nAcurácia no conjunto de teste: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Focado (0)', 'Procrastinou (1)']))

print("Pesos aprendidos por variável (Coeficientes):")
for feat, coef in zip(features, modelo.coef_[0]):
    print(f"  - {feat}: {coef:+.4f}")
print(f"  - Intercepto: {modelo.intercept_[0]:+.4f}")

# 6. Salvamento do modelo treinado
joblib.dump(modelo, 'modelo_procrastinacao.joblib')
print("\nModelo exportado com sucesso para 'modelo_procrastinacao.joblib'.")
print("=" * 60)
