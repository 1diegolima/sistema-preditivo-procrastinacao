"""
Passo 4 - Programe o Mínimo
Treinamento essencial de um modelo de Classificação (Regressão Logística)
para prever risco de procrastinação com poucas linhas de código.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Carregamento dos dados enxutos
df = pd.read_csv('dados_estudo.csv')

# 2. Definição das variáveis de entrada (X) e saída (y)
X = df[['horas_sono', 'minutos_rede_social', 'bloco_planejado_horas']]
y = df['procrastinou']

# 3. Treinamento do Modelo Mínimo
modelo = LogisticRegression()
modelo.fit(X, y)

# 4. Avaliação e verificação do funcionamento
acuracia = modelo.score(X, y)
print("=== Passo 4: Modelo Treinado com Sucesso ===")
print(f"Acurácia nos dados de treino: {acuracia * 100:.1f}%")
print(f"Pesos aprendidos (Coeficientes): {modelo.coef_[0]}")
print(f"Intercepto: {modelo.intercept_[0]:.4f}")

# 5. Teste rápido de inferência
exemplo_teste = pd.DataFrame([{
    'horas_sono': 7.0,
    'minutos_rede_social': 20,
    'bloco_planejado_horas': 2.0
}])

probabilidade = modelo.predict_proba(exemplo_teste)[0][1]
previsao = modelo.predict(exemplo_teste)[0]

print("\n--- Teste de Inferência Rápido ---")
print(f"Entrada de teste: 7h sono | 20m rede social | bloco de 2h")
print(f"Probabilidade de procrastinar: {probabilidade * 100:.2f}%")
print(f"Classificação final: {'Procrastina' if previsao == 1 else 'Foco Mantido'}")

# Salvar o modelo treinado para uso posterior no assistente
joblib.dump(modelo, 'modelo_procrastinacao.joblib')
print("\nModelo exportado para 'modelo_procrastinacao.joblib'.")
