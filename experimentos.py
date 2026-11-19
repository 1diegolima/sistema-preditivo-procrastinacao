"""
Passo 5 - Mude Uma Coisa por Vez
Execução de experimentos científicos rigorosamente controlados:
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

print("=" * 60)
print("     PASSO 5: EXECUTANDO EXPERIMENTOS CONTROLADOS")
print("=" * 60)

# -------------------------------------------------------------
# EXPERIMENTO 1: Variação exclusiva de Redes Sociais / Tela
# Fixos: Sono = 7.0h | Estudo Planejado = 2.5h | Atividade = 1.0h
# -------------------------------------------------------------
minutos_range = np.arange(0, 185, 5)
probs_exp1 = []

for m in minutos_range:
    cenario = pd.DataFrame([{
        'horas_sono': 7.0,
        'minutos_rede_social': m,
        'horas_estudo_planejado': 2.5,
        'horas_atividade_fisica': 1.0
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp1.append(p)

limiar_50_exp1 = None
for m, p in zip(minutos_range, probs_exp1):
    if p >= 50:
        limiar_50_exp1 = m
        break

plt.figure(figsize=(9, 5))
plt.plot(minutos_range, probs_exp1, color='#e63946', linewidth=2.5, label='Curva de Risco da IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')
if limiar_50_exp1:
    plt.axvline(x=limiar_50_exp1, color='#e76f51', linestyle=':', label=f'Ponto Crítico (~{limiar_50_exp1} min)')

plt.title('Experimento 1: Efeito do Tempo de Tela no Risco de Procrastinação\n(Fixos: 7h Sono | 2.5h Estudo | 1h Exercício)', fontsize=11, fontweight='bold')
plt.xlabel('Tempo em Redes Sociais antes da Sessão (minutos)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar (%)', fontsize=10)
plt.ylim(-5, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento1_tempo_tela.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# EXPERIMENTO 2: Variação exclusiva de Horas de Sono
# Fixos: Redes = 60 min | Estudo Planejado = 3.0h | Atividade = 0.5h
# -------------------------------------------------------------
sono_range = np.linspace(4.0, 9.5, 40)
probs_exp2 = []

for s in sono_range:
    cenario = pd.DataFrame([{
        'horas_sono': s,
        'minutos_rede_social': 60,
        'horas_estudo_planejado': 3.0,
        'horas_atividade_fisica': 0.5
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp2.append(p)

plt.figure(figsize=(9, 5))
plt.plot(sono_range, probs_exp2, color='#2a9d8f', linewidth=2.5, label='Curva de Risco da IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')
plt.title('Experimento 2: Efeito Protetor do Sono contra a Procrastinação\n(Fixos: 60 min Redes | 3h Estudo | 0.5h Exercício)', fontsize=11, fontweight='bold')
plt.xlabel('Horas de Sono na Noite Anterior (horas)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar (%)', fontsize=10)
plt.ylim(-5, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('graficos/experimento2_horas_sono.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# EXPERIMENTO 3: Variação da Carga de Estudo Planejada (Sobrecarga)
# Fixos: Sono = 7.0h | Redes = 45 min | Atividade = 0.5h
# -------------------------------------------------------------
estudo_range = np.linspace(1.0, 7.0, 30)
probs_exp3 = []

for e in estudo_range:
    cenario = pd.DataFrame([{
        'horas_sono': 7.0,
        'minutos_rede_social': 45,
        'horas_estudo_planejado': e,
        'horas_atividade_fisica': 0.5
    }])
    p = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp3.append(p)

plt.figure(figsize=(9, 5))
plt.plot(estudo_range, probs_exp3, color='#f4a261', linewidth=2.5, label='Curva de Risco da IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')
plt.title('Experimento 3: Sobrecarga Mental - Duração do Bloco sem Pausas\n(Fixos: 7h Sono | 45 min Redes | 0.5h Exercício)', fontsize=11, fontweight='bold')
plt.xlabel('Duração do Bloco de Estudo Planejado (horas contínuas)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar (%)', fontsize=10)
plt.ylim(-5, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento3_sobrecarga_estudo.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# GRÁFICO 4: Comparativo de Coeficientes (Pesos Aprendidos)
# -------------------------------------------------------------
nomes_features = ['Horas de Sono', 'Minutos de Rede Social', 'Estudo Planejado (h)', 'Atividade Física (h)']
coefs = modelo.coef_[0]
cores = ['#2a9d8f' if c < 0 else '#e63946' for c in coefs]

plt.figure(figsize=(9, 5))
bars = plt.barh(nomes_features, coefs, color=cores)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Passo 6: Pesos Aprendidos pela Regressão Logística\n(Verde = Protege contra Procrastinação | Vermelho = Aumenta Risco)', fontsize=11, fontweight='bold')
plt.xlabel('Magnitude do Coeficiente Linear', fontsize=10)

for bar in bars:
    w = bar.get_width()
    pos_x = w + (0.02 if w >= 0 else -0.02)
    ha = 'left' if w >= 0 else 'right'
    plt.text(pos_x, bar.get_y() + bar.get_height()/2, f'{w:+.4f}', va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos/importancia_fatores.png', dpi=300)
plt.close()

print(f"Experimento 1 concluído. Ponto crítico de tela: ~{limiar_50_exp1} minutos.")
print("Todos os gráficos salvos com sucesso na pasta 'graficos/'.")
print("=" * 60)
