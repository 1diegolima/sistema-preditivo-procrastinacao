"""
Passo 4 - Programe o Mínimo: Treinamento essencial da IA
Problema Real: Previsão de Procrastinação e Perda de Foco em Estudantes
Algoritmo: Regressão Logística (Supervisionado - Classificação Probabilística)
Base de Dados: dataset_estudantes_real.csv (481 respostas de estudantes reais)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Carregamento dos dados reais
df = pd.read_csv('dataset_estudantes_real.csv')

# 2. Definição das variáveis preditoras (X) e da variável alvo (y)
features = [
    'horas_redes_sociais',       # Tempo diário em telas/redes (horas)
    'busca_dopamina_scroll',     # Frequência de uso sem propósito/busca de dopamina (1-5)
    'ansiedade_sem_celular',     # Inquietação/abstinência sem redes (1-5)
    'problemas_sono',            # Distúrbios e insônia (1-5)
    'dificuldade_concentracao'   # Dificuldade autorreferida em focar (1-5)
]

X = df[features]
y = df['procrastina_ao_estudar']  # 1 = Procrastina/distrai-se quando ocupado; 0 = Mantém o foco

# 3. Divisão em Treino (80%) e Teste (20%) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Treinamento do Modelo Mínimo (Regressão Logística)
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train, y_train)

# 5. Avaliação do Modelo no conjunto de teste independente
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("=" * 65)
print("     PASSO 4: MODELO DE INTELIGÊNCIA ARTIFICIAL TREINADO COM SUCESSO")
print("=" * 65)
print(f"Total de registros reais na base: {len(df)}")
print(f"Amostras de treino: {len(X_train)} (80%) | Amostras de teste: {len(X_test)} (20%)")
print(f"\nAcurácia no conjunto de teste: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Focado (0)', 'Procrastinou (1)']))

print("Matriz de Confusão:")
matriz = confusion_matrix(y_test, y_pred)
print(f"  Verdadeiros Negativos (Foco correto): {matriz[0][0]}")
print(f"  Falsos Positivos (Alerta falso):     {matriz[0][1]}")
print(f"  Falsos Negativos (Não detectou):     {matriz[1][0]}")
print(f"  Verdadeiros Positivos (Detectou):    {matriz[1][1]}")

print("\nPesos Aprendidos por Fator (Coeficientes da Regressão Logística):")
for feat, coef in zip(features, modelo.coef_[0]):
    print(f"  - {feat:25s}: {coef:+.4f}")
print(f"  - {'Intercepto basal':25s}: {modelo.intercept_[0]:+.4f}")

# 6. Salvamento do modelo treinado para inferência
joblib.dump(modelo, 'modelo_procrastinacao.joblib')
print("\nModelo serializado salvo com sucesso em 'modelo_procrastinacao.joblib'.")
print("=" * 65)
