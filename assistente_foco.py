"""
Assistente Interativo de Foco e Produtividade (Terminal Bot)
Utiliza o modelo treinado de Machine Learning para diagnosticar o risco
de procrastinação de um estudante e fornecer orientações personalizadas.
"""

import sys
import joblib
import pandas as pd

def carregar_modelo():
    try:
        return joblib.load('modelo_procrastinacao.joblib')
    except Exception as e:
        print("\n[ERRO] Modelo 'modelo_procrastinacao.joblib' não encontrado.")
        print("Execute primeiro 'python treinar_modelo.py' para treinar a IA.")
        sys.exit(1)

def diagnosticar(modelo, sono, redes, bloco):
    entrada = pd.DataFrame([{
        'horas_sono': sono,
        'minutos_rede_social': redes,
        'bloco_planejado_horas': bloco
    }])
    
    prob_procrastinar = modelo.predict_proba(entrada)[0][1] * 100
    classificacao = modelo.predict(entrada)[0]
    return prob_procrastinar, classificacao

def main():
    modelo = carregar_modelo()
    
    print("\n" + "=" * 60)
    print("      🧠 ASSISTENTE INTELIGENTE DE FOCO & PRODUTIVIDADE")
    print("           (Módulo Preditivo Anti-Procrastinação)")
    print("=" * 60)
    print("Responda às 3 perguntas para diagnosticar sua sessão de estudos:\n")
    
    try:
        sono = float(input("1. Quantas horas você dormiu na noite passada? (ex: 7.0): "))
        redes = float(input("2. Quantos minutos passou em redes sociais antes de estudar? (ex: 35): "))
        bloco = float(input("3. Quantas horas pretende estudar sem interrupções? (ex: 2.0): "))
    except ValueError:
        print("\n[ERRO] Por favor, digite números válidos.")
        return

    prob, decisao = diagnosticar(modelo, sono, redes, bloco)
    
    print("\n" + "-" * 60)
    print(f"📊 DIAGNÓSTICO DO MODELO DE IA:")
    print(f"   -> Risco calculado de Procrastinação: {prob:.1f}%")
    print("-" * 60)
    
    if prob < 35:
        print("🟢 STATUS: FOCO BLINDADO (Baixo Risco)")
        print("💡 Feedback da IA: Suas condições biológicas e atencionais estão ótimas!")
        print("   Recomendação: Mantenha o ambiente silencioso e comece sua sessão.")
    elif prob <= 65:
        print("🟡 STATUS: ATENÇÃO MODERADA (Zona de Transição)")
        print("💡 Feedback da IA: Você está próximo do limiar de dispersão.")
        print("   Recomendação: Deixe o celular em outro cômodo e planeje uma pausa curta a cada 45 minutos.")
    else:
        print("🔴 STATUS: ALERTA CRÍTICO (Alto Risco de Procrastinação)")
        print("💡 Feedback da IA: O excesso de estímulos imediatos e/ou cansaço reduziram seu autocontrole.")
        print("   Recomendação:")
        print("   - Reduza a meta para um bloco menor de 25 minutos (Pomodoro).")
        print("   - Tome um copo de água e afaste-se de qualquer tela por 10 minutos antes de recomeçar.")

    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
