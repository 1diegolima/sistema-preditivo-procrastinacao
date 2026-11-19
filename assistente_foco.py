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
    except Exception:
        print("\n[ERRO] Modelo 'modelo_procrastinacao.joblib' não encontrado.")
        print("Execute primeiro 'python treinar_modelo.py' para treinar a IA.")
        sys.exit(1)

def diagnosticar(modelo, sono, redes, estudo, atividade):
    entrada = pd.DataFrame([{
        'horas_sono': sono,
        'minutos_rede_social': redes,
        'horas_estudo_planejado': estudo,
        'horas_atividade_fisica': atividade
    }])
    
    prob_procrastinar = modelo.predict_proba(entrada)[0][1] * 100
    classificacao = modelo.predict(entrada)[0]
    return prob_procrastinar, classificacao

def main():
    modelo = carregar_modelo()
    
    print("\n" + "=" * 65)
    print("      🧠 ASSISTENTE INTELIGENTE DE FOCO & PRODUTIVIDADE")
    print("           (Módulo Preditivo Anti-Procrastinação)")
    print("=" * 65)
    print("Responda às 4 perguntas rápidas sobre a sua rotina de hoje:\n")
    
    try:
        sono = float(input("1. Quantas horas você dormiu na noite passada? (ex: 7.0): "))
        redes = float(input("2. Quantos minutos passou em redes sociais antes de estudar? (ex: 45): "))
        estudo = float(input("3. Quantas horas pretende estudar sem interrupções? (ex: 2.5): "))
        atividade = float(input("4. Quantas horas de atividade física fez/fará hoje? (ex: 1.0 ou 0): "))
    except ValueError:
        print("\n[ERRO] Entrada inválida. Por favor, digite números válidos.")
        return

    prob, decisao = diagnosticar(modelo, sono, redes, estudo, atividade)
    
    print("\n" + "-" * 65)
    print(f"📊 DIAGNÓSTICO DO MODELO DE INTELIGÊNCIA ARTIFICIAL:")
    print(f"   -> Risco calculado de Procrastinação: {prob:.1f}%")
    print("-" * 65)
    
    if prob < 35:
        print("🟢 STATUS: FOCO BLINDADO (Baixo Risco)")
        print("💡 Feedback da IA: Suas condições biológicas e atencionais estão ótimas!")
        print("   Recomendação: Mantenha o celular distante e inicie sua sessão com confiança.")
    elif prob <= 65:
        print("🟡 STATUS: ATENÇÃO MODERADA (Zona de Transição)")
        print("💡 Feedback da IA: Você está na fronteira da dispersão mental.")
        print("   Recomendação:")
        print("   - Divida seu bloco em ciclos de 45 minutos com 10 minutos de descanso.")
        print("   - Coloque o celular em modo foco ou em outro cômodo.")
    else:
        print("🔴 STATUS: ALERTA CRÍTICO (Alto Risco de Procrastinação)")
        print("💡 Feedback da IA: O estímulo prévio de telas ou cansaço reduziram seu autocontrole.")
        print("   Recomendação:")
        print("   - Reduza a meta para um bloco menor de 25 minutos (Técnica Pomodoro).")
        print("   - Tome um copo de água e faça 5 minutos de respiração longe de telas antes de abrir os livros.")

    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
