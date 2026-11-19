"""
Passo 5 - Mude Uma Coisa por Vez
Realiza experimentos científicos controlados alterando estritamente uma única variável
de entrada enquanto fixa as outras, provando que a IA aprendeu relações lógicas.
Gera os gráficos que servirão de base para o Passo 6 (Explicação dos Resultados).
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cria diretório para salvar os gráficos
os.makedirs('graficos', exist_ok=True)

# 2. Carrega o modelo treinado no Passo 4
modelo = joblib.load('modelo_procrastinacao.joblib')

print("=== Passo 5: Iniciando Experimentos Controlados ===")

# -------------------------------------------------------------
# EXPERIMENTO 1: Variação exclusiva dos minutos de Redes Sociais
# Fixos: Sono = 7.0 horas, Bloco = 2.0 horas
# -------------------------------------------------------------
minutos_range = np.arange(0, 125, 5)
probs_exp1 = []

for m in minutos_range:
    cenario = pd.DataFrame([{
        'horas_sono': 7.0,
        'minutos_rede_social': m,
        'bloco_planejado_horas': 2.0
    }])
    prob = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp1.append(prob)

# Encontrar ponto onde passa de 50%
limiar_50 = None
for m, p in zip(minutos_range, probs_exp1):
    if p >= 50:
        limiar_50 = m
        break

print(f"[Experimento 1] Fixando Sono (7h) e Bloco (2h):")
print(f" -> 15 min de rede social: {probs_exp1[3]:.1f}% de risco")
print(f" -> 50 min de rede social: {probs_exp1[10]:.1f}% de risco")
print(f" -> 80 min de rede social: {probs_exp1[16]:.1f}% de risco")
if limiar_50:
    print(f" -> Limiar crítico de 50% atingido aos: {limiar_50} minutos de tela.")

# Gráfico do Experimento 1
plt.figure(figsize=(9, 5))
plt.plot(minutos_range, probs_exp1, color='#e63946', linewidth=2.5, label='Curva de Probabilidade da IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')
if limiar_50:
    plt.axvline(x=limiar_50, color='#f4a261', linestyle=':', label=f'Ponto Crítico (~{limiar_50} min)')

plt.title('Experimento 1: Efeito do Tempo de Tela no Risco de Procrastinação\n(Fixos: 7h de sono e 2h de bloco de estudo)', fontsize=12, fontweight='bold')
plt.xlabel('Tempo em Redes Sociais antes do Estudo (minutos)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar (%)', fontsize=10)
plt.ylim(-5, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('graficos/experimento1_tempo_tela.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# EXPERIMENTO 2: Variação exclusiva das Horas de Sono
# Fixos: Redes Sociais = 55 min, Bloco = 2.5 horas
# -------------------------------------------------------------
sono_range = np.linspace(4.0, 9.0, 30)
probs_exp2 = []

for s in sono_range:
    cenario = pd.DataFrame([{
        'horas_sono': s,
        'minutos_rede_social': 55,
        'bloco_planejado_horas': 2.5
    }])
    prob = modelo.predict_proba(cenario)[0][1] * 100
    probs_exp2.append(prob)

print(f"\n[Experimento 2] Fixando Redes (55 min) e Bloco (2.5h):")
print(f" -> 4.5h sono: {probs_exp2[3]:.1f}% de risco")
print(f" -> 7.0h sono: {probs_exp2[18]:.1f}% de risco")
print(f" -> 8.5h sono: {probs_exp2[27]:.1f}% de risco")

# Gráfico do Experimento 2
plt.figure(figsize=(9, 5))
plt.plot(sono_range, probs_exp2, color='#2a9d8f', linewidth=2.5, label='Curva de Probabilidade da IA')
plt.axhline(y=50, color='#6c757d', linestyle='--', label='Limiar de Decisão (50%)')
plt.title('Experimento 2: Efeito Protetor do Sono contra a Procrastinação\n(Fixos: 55 min de rede social e 2.5h de bloco)', fontsize=12, fontweight='bold')
plt.xlabel('Horas de Sono na Noite Anterior (horas)', fontsize=10)
plt.ylabel('Probabilidade de Procrastinar (%)', fontsize=10)
plt.ylim(-5, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('graficos/experimento2_horas_sono.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# GRÁFICO 3: Importância dos Coeficientes / Pesos do Modelo
# -------------------------------------------------------------
coefs = modelo.coef_[0]
features = ['Horas de Sono', 'Minutos de Rede Social', 'Duração do Bloco (h)']
cores = ['#2a9d8f' if c < 0 else '#e63946' for c in coefs]

plt.figure(figsize=(8, 4.5))
bars = plt.barh(features, coefs, color=cores)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Passo 6: Pesos Aprendidos pelo Modelo (Coeficientes da Regressão)', fontsize=12, fontweight='bold')
plt.xlabel('Magnitude e Direção do Coeficiente (Verde = Reduz Risco | Vermelho = Aumenta Risco)', fontsize=9)
for bar in bars:
    width = bar.get_width()
    ha = 'left' if width >= 0 else 'right'
    plt.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos/importancia_fatores.png', dpi=300)
plt.close()

print("\n=== Todos os experimentos foram concluídos com sucesso! ===")
print("Gráficos salvos em 'graficos/':")
print("  - experimento1_tempo_tela.png")
print("  - experimento2_horas_sono.png")
print("  - importancia_fatores.png")
