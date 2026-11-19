"""
Passo 5 - Mude Uma Coisa por Vez
Execução de experimentos controlados no modelo treinado com dados reais:
Altera apenas uma variável de entrada por vez mantendo todas as outras fixas.
Gera os gráficos explicativos para o Passo 6.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('graficos', exist_ok=True)
modelo = joblib.load('modelo_procrastinacao.joblib')

print("=" * 65)
print("     PASSO 5: EXECUTANDO EXPERIMENTOS CIENTÍFICOS CONTROLADOS")
print("=" * 65)

# Configuração de estilo geral para os gráficos
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# EXPERIMENTO 1: Variação exclusiva das Horas de Redes Sociais
# Fixos: Dopamina Scroll = 3 | Ansiedade = 2 | Sono = 2 | Concentração = 2
# -------------------------------------------------------------
horas_range = np.linspace(0.0, 7.0, 50)
probs_exp1 = []

for h in horas_range:
    cenario = pd.DataFrame([{
        'horas_redes_sociais': h,
        'busca_dopamina_scroll': 3,
        'ansiedade_sem_celular': 2,
        'problemas_sono': 2,
        'dificuldade_concentracao': 2
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp1.append(p)

plt.figure(figsize=(9, 5))
plt.plot(horas_range, probs_exp1, color='#e63946', linewidth=2.5, label='Probabilidade Prevista pela IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')

plt.title('Experimento 1: Efeito do Tempo Diário em Redes Sociais no Risco de Procrastinação\n(Fixos: Busca Dopamina=3 | Ansiedade=2 | Problemas Sono=2 | Dificuldade Foco=2)', fontsize=10, fontweight='bold')
plt.xlabel('Tempo Diário em Redes Sociais (horas/dia)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar ao Estudar (%)', fontsize=10)
plt.ylim(0, 100)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento1_tempo_tela.png', dpi=300)
plt.close()
print("Experimento 1 concluído -> 'graficos/experimento1_tempo_tela.png'")

# -------------------------------------------------------------
# EXPERIMENTO 2: Variação exclusiva da Busca por Dopamina / Scroll sem Propósito
# Fixos: Redes = 2.5h | Ansiedade = 2 | Sono = 2 | Concentração = 2
# -------------------------------------------------------------
scroll_range = np.linspace(1.0, 5.0, 40)
probs_exp2 = []

for sc in scroll_range:
    cenario = pd.DataFrame([{
        'horas_redes_sociais': 2.5,
        'busca_dopamina_scroll': sc,
        'ansiedade_sem_celular': 2,
        'problemas_sono': 2,
        'dificuldade_concentracao': 2
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp2.append(p)

plt.figure(figsize=(9, 5))
plt.plot(scroll_range, probs_exp2, color='#457b9d', linewidth=2.5, label='Probabilidade Prevista pela IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')

plt.title('Experimento 2: Efeito da Busca por Dopamina (Scroll Involuntário sem Propósito)\n(Fixos: Redes Sociais=2.5h | Ansiedade=2 | Problemas Sono=2 | Dificuldade Foco=2)', fontsize=10, fontweight='bold')
plt.xlabel('Frequência de Scroll Involuntário / Sem Propósito (Escala 1 a 5)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar ao Estudar (%)', fontsize=10)
plt.ylim(0, 100)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento2_busca_dopamina.png', dpi=300)
plt.close()
print("Experimento 2 concluído -> 'graficos/experimento2_busca_dopamina.png'")

# -------------------------------------------------------------
# EXPERIMENTO 3: Variação da Ansiedade e Inquietação sem Celular (Abstinência)
# Fixos: Redes = 2.5h | Dopamina Scroll = 3 | Sono = 2 | Concentração = 2
# -------------------------------------------------------------
ansiedade_range = np.linspace(1.0, 5.0, 40)
probs_exp3 = []

for a in ansiedade_range:
    cenario = pd.DataFrame([{
        'horas_redes_sociais': 2.5,
        'busca_dopamina_scroll': 3,
        'ansiedade_sem_celular': a,
        'problemas_sono': 2,
        'dificuldade_concentracao': 2
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp3.append(p)

plt.figure(figsize=(9, 5))
plt.plot(ansiedade_range, probs_exp3, color='#e76f51', linewidth=2.5, label='Probabilidade Prevista pela IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')

plt.title('Experimento 3: Efeito da Ansiedade/Inquietação ao Ficar Longe do Celular\n(Fixos: Redes Sociais=2.5h | Busca Dopamina=3 | Problemas Sono=2 | Dificuldade Foco=2)', fontsize=10, fontweight='bold')
plt.xlabel('Inquietação / Ansiedade ao Ficar Sem Redes Sociais (Escala 1 a 5)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar ao Estudar (%)', fontsize=10)
plt.ylim(0, 100)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento3_ansiedade_celular.png', dpi=300)
plt.close()
print("Experimento 3 concluído -> 'graficos/experimento3_ansiedade_celular.png'")

# -------------------------------------------------------------
# GRÁFICO 4 (Passo 6): Comparativo dos Coeficientes (Pesos Aprendidos)
# -------------------------------------------------------------
features_labels = [
    'Inquietação sem Celular\n(Abstinência de Dopamina)',
    'Dificuldade de\nConcentração Prévia',
    'Scroll sem Propósito\n(Hábito Involuntário)',
    'Problemas de Sono\n(Fadiga / Insônia)',
    'Horas em Redes\nSociais por Dia'
]

# Ordenar coeficientes por magnitude
coefs_dict = {
    'Inquietação sem Celular\n(Abstinência de Dopamina)': modelo.coef_[0][2],
    'Dificuldade de\nConcentração Prévia': modelo.coef_[0][4],
    'Scroll sem Propósito\n(Hábito Involuntário)': modelo.coef_[0][1],
    'Problemas de Sono\n(Fadiga / Insônia)': modelo.coef_[0][3],
    'Horas em Redes\nSociais por Dia': modelo.coef_[0][0]
}

sorted_items = sorted(coefs_dict.items(), key=lambda x: x[1])
labels = [item[0] for item in sorted_items]
valores = [item[1] for item in sorted_items]

plt.figure(figsize=(9, 5))
bars = plt.barh(labels, valores, color='#2a9d8f')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Passo 6: Pesos dos Fatores Aprendidos pela IA na Base de Dados Real\n(Coeficientes da Regressão Logística - Quanto maior, maior o risco de procrastinar)', fontsize=10, fontweight='bold')
plt.xlabel('Magnitude do Impacto no Log-Odds de Procrastinar (+)', fontsize=10)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.015, bar.get_y() + bar.get_height()/2, f'+{w:.4f}', va='center', ha='left', fontsize=10, fontweight='bold', color='#1d3557')

plt.xlim(0, max(valores) + 0.1)
plt.tight_layout()
plt.savefig('graficos/importancia_fatores.png', dpi=300)
plt.close()
print("Gráfico de Importância concluído -> 'graficos/importancia_fatores.png'")

# -------------------------------------------------------------
# GRÁFICO 5 (Passo 6): Matriz de Confusão no Conjunto de Teste
# -------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df_real = pd.read_csv('dataset_estudantes_real.csv')
features_exp = ['horas_redes_sociais', 'busca_dopamina_scroll', 'ansiedade_sem_celular', 'problemas_sono', 'dificuldade_concentracao']
X_all = df_real[features_exp]
y_all = df_real['procrastina_ao_estudar']

_, X_test_cm, _, y_test_cm = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)
y_pred_cm = modelo.predict(X_test_cm)
cm = confusion_matrix(y_test_cm, y_pred_cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Passo 6: Matriz de Confusão (Conjunto de Teste - 97 Casos Reais)', fontsize=10, fontweight='bold')
plt.colorbar(fraction=0.046, pad=0.04)

classes = ['Foco Preservado (0)', 'Procrastinou (1)']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=9)
plt.yticks(tick_marks, classes, fontsize=9, rotation=45)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')

plt.ylabel('Rótulo Real', fontsize=10)
plt.xlabel('Previsão da IA', fontsize=10)
plt.tight_layout()
plt.savefig('graficos/matriz_confusao.png', dpi=300)
plt.close()
print("Gráfico Matriz de Confusão concluído -> 'graficos/matriz_confusao.png'")

print("=" * 65)
print("Todos os experimentos e gráficos foram gerados com sucesso!")
print("=" * 65)
