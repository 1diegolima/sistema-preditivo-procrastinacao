"""
Assistente Interativo de Foco e Autodiagnóstico de Procrastinação
Demonstração Prática do Modelo de IA treinado com Dados Reais
"""

import joblib
import pandas as pd

def coletar_resposta(pergunta, min_val, max_val, tipo=int):
    while True:
        try:
            val = tipo(input(f"{pergunta} [{min_val} a {max_val}]: "))
            if min_val <= val <= max_val:
                return val
            print(f"Por favor, digite um valor entre {min_val} e {max_val}.")
        except ValueError:
            print(f"Entrada inválida. Digite um número válido.")

def main():
    print("=" * 65)
    print("   🧠 ASSISTENTE INTELIGENTE: PREVISÃO DE FOCO E PROCRASTINAÇÃO")
    print("   Baseado em Inteligência Artificial treinada com Dados Reais")
    print("=" * 65)
    print("Responda às 5 perguntas abaixo para avaliar seu risco de dispersão:\n")

    # Coleta das entradas
    horas_redes = coletar_resposta(
        "1. Quantas horas por dia você costuma passar no celular/redes sociais?",
        0.0, 16.0, float
    )
    
    busca_dopamina = coletar_resposta(
        "2. Com que frequência você abre redes sociais no automático / sem propósito (busca de dopamina)? (1=Raramente, 5=O tempo todo)",
        1, 5, int
    )
    
    ansiedade = coletar_resposta(
        "3. Você sente inquietação/ansiedade quando passa um tempo sem checar o celular? (1=Nenhuma, 5=Muito intensa)",
        1, 5, int
    )
    
    sono = coletar_resposta(
        "4. Com que frequência você tem problemas de sono / insônia? (1=Raramente, 5=Quase toda noite)",
        1, 5, int
    )
    
    concentracao = coletar_resposta(
        "5. No geral, quão difícil é para você sustentar a concentração em tarefas densas? (1=Muito fácil, 5=Muito difícil)",
        1, 5, int
    )

    # Montagem do DataFrame para o modelo
    perfil = pd.DataFrame([{
        'horas_redes_sociais': horas_redes,
        'busca_dopamina_scroll': busca_dopamina,
        'ansiedade_sem_celular': ansiedade,
        'problemas_sono': sono,
        'dificuldade_concentracao': concentracao
    }])

    # Carregamento e Inferência
    try:
        modelo = joblib.load('modelo_procrastinacao.joblib')
        probabilidade = modelo.predict_proba(perfil)[0][1] * 100
        decisao = modelo.predict(perfil)[0]
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    print("\n" + "=" * 65)
    print("                      DIAGNÓSTICO DA IA")
    print("=" * 65)
    print(f"Probabilidade Estimada de Dispersão/Procrastinação: {probabilidade:.1f}%")

    if decisao == 1:
        print("Status: 🔴 ALTO RISCO DE PROCRASTINAÇÃO E PERDA DE FOCO")
        print("\nRecomendações Práticas Imediatas:")
        if ansiedade >= 4 or busca_dopamina >= 4:
            print("  ⚡ Reduza os Gatilhos de Dopamina: Guarde o celular em outro cômodo durante os blocos de estudo.")
            print("  ⚡ Adote Fricção Digital: Desative notificações ou ative a tela em escala de cinza.")
        if horas_redes >= 4.0:
            print("  ⚡ Limite de Uso: Estabeleça um teto diário de tela para proteger seu córtex pré-frontal da saturação.")
        if sono >= 4:
            print("  ⚡ Higiene do Sono: Evite telas pelo menos 45 minutos antes de dormir para restaurar a energia inibitória.")
        print("  ⚡ Técnica Pomodoro: Inicie com ciclos curtos (25 min de foco estrito / 5 min de pausa sem telas).")
    else:
        print("Status: 🟢 PERFIL DE ALTO FOCO / BAIXA PROPENSÃO À PROCRASTINAÇÃO")
        print("\nParabéns! Seus hábitos de uso de tela e autorregulação protegem sua capacidade atencional.")
        print("Dica: Mantenha as pausas ativas longe de feeds infinitos para preservar o estado de flow.")
    
    print("=" * 65)

if __name__ == '__main__':
    main()
