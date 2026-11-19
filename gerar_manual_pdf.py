import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# Canvas Personalizado com Paginação Automática ("Página X de Y")
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        
        # Ignora header e footer na capa (página 1)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 800, "Sistema Preditivo de Foco & Combate à Procrastinação • Guia Técnico de Código")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 794, 541, 794)

            # Footer
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 36, "Aplicação do Roteiro de 6 Passos de IA • Prof. Wagner")
            page_text = f"Página {self._pageNumber} de {total_pages}"
            self.drawRightString(541, 36, page_text)
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 46, 541, 46)

        self.restoreState()

# ----------------------------------------------------------------------
# Gerador Principal do Documento
# ----------------------------------------------------------------------
def gerar_pdf(caminho_saida="Manual_Explicativo_Codigo_IA_Wagner.pdf"):
    doc = SimpleDocTemplate(
        caminho_saida,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Estilos customizados
    c_primary = colors.HexColor("#0f172a")     # Slate 900
    c_accent = colors.HexColor("#0284c7")      # Sky 600
    c_success = colors.HexColor("#059669")     # Emerald 600
    c_code_bg = colors.HexColor("#f8fafc")     # Slate 50
    c_code_border = colors.HexColor("#cbd5e1") # Slate 300

    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=c_primary,
        spaceAfter=8
    )

    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceAfter=20
    )

    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=20,
        textColor=c_primary,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    style_body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=style_body,
        fontName='Helvetica-Bold'
    )

    style_code = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    style_callout = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # ==================================================================
    # CAPA / CABEÇALHO DO DOCUMENTO
    # ==================================================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("SISTEMA PREDITIVO DE FOCO & COMBATE À PROCRASTINAÇÃO", style_title))
    story.append(Paragraph("Guia Técnico do Código & Defesa do Roteiro de 6 Passos de Inteligência Artificial", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceBefore=0, spaceAfter=15))

    meta_data = [
        [Paragraph("<b>Projeto:</b>", style_body), Paragraph("Sistema Preditivo com Base Real, Web App Minimalista, Pomodoro e Gemini", style_body)],
        [Paragraph("<b>Metodologia:</b>", style_body), Paragraph("Roteiro Prático de 6 Passos de Inteligência Artificial", style_body)],
        [Paragraph("<b>Professor:</b>", style_body), Paragraph("Wagner", style_body)],
        [Paragraph("<b>Finalidade:</b>", style_body), Paragraph("Manual de consulta para explicar e defender cada linha de código", style_body)]
    ]
    t_meta = Table(meta_data, colWidths=[90, 397])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # ==================================================================
    # SEÇÃO 1: VISÃO GERAL DA ARQUITETURA E MAPA DE ARQUIVOS
    # ==================================================================
    story.append(Paragraph("1. Visão Geral da Arquitetura do Sistema", style_h1))
    story.append(Paragraph(
        "Este projeto foi estruturado para atender com rigor metodológico aos <b>6 Passos de IA</b> solicitados pelo professor Wagner. "
        "Diferente de projetos acadêmicos genéricos que utilizam dados simulados ou bibliotecas automáticas (AutoML), "
        "aqui cada etapa foi construída manualmente, utilizando uma base comportamental real de 481 estudantes e implementando uma "
        "<b>Arquitetura Híbrida</b> (Machine Learning Clássico interpretável + IA Generativa no Coach Zen + Aplicação Web com Pomodoro).",
        style_body
    ))

    # Tabela com mapa dos arquivos
    story.append(Spacer(1, 6))
    files_table_data = [
        [Paragraph("<b>Arquivo</b>", style_body_bold), Paragraph("<b>Passo / Função</b>", style_body_bold), Paragraph("<b>O que faz</b>", style_body_bold)],
        [Paragraph("<code>dataset_estudantes_real.csv</code>", style_code), Paragraph("Passo 1 (Problema)", style_body), Paragraph("Base de dados com 481 registros reais contendo variáveis de tela, sono e foco.", style_body)],
        [Paragraph("<code>dados_estudo.csv</code>", style_code), Paragraph("Passo 3 (Exemplo Pequeno)", style_body), Paragraph("Piloto contrastante com 10 casos reais para validar o raciocínio manualmente.", style_body)],
        [Paragraph("<code>treinar_modelo.py</code>", style_code), Paragraph("Passo 4 (Programar Mínimo)", style_body), Paragraph("Carrega dados, divide 80/20 estratificado, treina Regressão Logística e exporta .joblib.", style_body)],
        [Paragraph("<code>experimentos.py</code>", style_code), Paragraph("Passo 5 (Mudar 1 Coisa)", style_body), Paragraph("Executa 3 testes controlados congelando variáveis e gera 5 gráficos em <code>graficos/</code>.", style_body)],
        [Paragraph("<code>assistente_foco.py</code>", style_code), Paragraph("Inferência Terminal", style_body), Paragraph("Coleta 5 inputs pelo console e exibe o risco de procrastinação com dicas imediatas.", style_body)],
        [Paragraph("<code>app.py</code>", style_code), Paragraph("Produto Web Completo", style_body), Paragraph("Servidor web nativo (porta 5000) com Diagnóstico, Pomodoro, Coach Gemini, Atividades e Relatório.", style_body)]
    ]
    t_files = Table(files_table_data, colWidths=[120, 110, 257])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_files)
    story.append(Spacer(1, 15))

    # ==================================================================
    # SEÇÃO 2: A BASE DE DADOS REAL E AS 5 VARIÁVEIS
    # ==================================================================
    story.append(Paragraph("2. A Base de Dados Real (Passo 1 do Roteiro)", style_h1))
    story.append(Paragraph(
        "Os dados são provenientes do estudo público <i>Social Media Usage and Mental Health Survey</i>, estruturado no arquivo "
        "<code>dataset_estudantes_real.csv</code> com <b>481 observações reais</b>. Cada linha representa um estudante com 5 variáveis preditoras (X) "
        "e 1 variável alvo binária (y):",
        style_body
    ))

    vars_data = [
        [Paragraph("<b>Variável</b>", style_body_bold), Paragraph("<b>Tipo / Escala</b>", style_body_bold), Paragraph("<b>Significado Neurocomportamental</b>", style_body_bold)],
        [Paragraph("<code>horas_redes_sociais</code>", style_code), Paragraph("Numérica Contínua (horas/dia)", style_body), Paragraph("Exposição bruta diária a telas. Causa sobrecarga cumulativa na atenção.", style_body)],
        [Paragraph("<code>busca_dopamina_scroll</code>", style_code), Paragraph("Ordinal (1 a 5)", style_body), Paragraph("Frequência de abertura compulsiva de redes sem propósito específico (busca de estímulo rápido).", style_body)],
        [Paragraph("<code>ansiedade_sem_celular</code>", style_code), Paragraph("Ordinal (1 a 5)", style_body), Paragraph("Sintoma de abstinência e inquietação ao ficar longe do smartphone (fator de maior peso no modelo).", style_body)],
        [Paragraph("<code>problemas_sono</code>", style_code), Paragraph("Ordinal (1 a 5)", style_body), Paragraph("Privação de descanso, que enfraquece a capacidade inibitória do córtex pré-frontal.", style_body)],
        [Paragraph("<code>dificuldade_concentracao</code>", style_code), Paragraph("Ordinal (1 a 5)", style_body), Paragraph("Vulnerabilidade atencional prévia e facilidade de dispersão em tarefas densas.", style_body)],
        [Paragraph("<code>procrastina_ao_estudar</code> (y)", style_code), Paragraph("Binária (0 ou 1)", style_body), Paragraph("<b>0 = Foco Mantido</b> (baixa dispersão) | <b>1 = Procrastinou / Evasão</b> (perdeu o foco de estudo).", style_body)]
    ]
    t_vars = Table(vars_data, colWidths=[130, 110, 247])
    t_vars.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284c7")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_vars)

    # ==================================================================
    # SEÇÃO 3: TREINAMENTO DO MODELO - treinar_modelo.py
    # ==================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Código de Treinamento da IA (<code>treinar_modelo.py</code>)", style_h1))
    story.append(Paragraph(
        "Este script corresponde ao <b>Passo 4 (Programe o Mínimo)</b>. Ele realiza o fluxo canônico de Aprendizado Supervisionado "
        "em apenas 67 linhas, sem complexidade desnecessária. Veja o que cada bloco executa:",
        style_body
    ))

    code_treino_1 = """# Divisão Estratificada em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)"""
    t_c1 = Table([[Paragraph(code_treino_1.replace("\n", "<br/>"), style_code)]], colWidths=[487])
    t_c1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_code_bg),
        ('BOX', (0,0), (-1,-1), 1, c_code_border),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_c1)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Por que o <code>stratify=y</code> é importante?</b> Ele garante que a proporção exata entre focado (0) e procrastinador (1) "
        "seja idêntica no conjunto de treino (384 amostras) e no de teste (97 amostras), evitando que o modelo seja avaliado em um teste desbalanceado.",
        style_body
    ))

    story.append(Spacer(1, 8))
    code_treino_2 = """# Treinamento da Regressão Logística
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train, y_train)

# Avaliação no conjunto de teste nunca visto
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred) # -> 68.04%
joblib.dump(modelo, 'modelo_procrastinacao.joblib')"""
    t_c2 = Table([[Paragraph(code_treino_2.replace("\n", "<br/>"), style_code)]], colWidths=[487])
    t_c2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_code_bg),
        ('BOX', (0,0), (-1,-1), 1, c_code_border),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_c2)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>O que o <code>modelo.fit()</code> calcula matematicamente?</b> Ele ajusta os coeficientes beta da função sigmoide "
        "através da maximização da verossimilhança (Maximum Likelihood Estimation), encontrando o peso que minimiza o erro de previsão. "
        "Em seguida, <code>joblib.dump()</code> serializa o modelo treinado em disco para que a aplicação web e o assistente usem sem retreinar.",
        style_body
    ))

    story.append(Paragraph("Pesos Finais Aprendidos pelo Modelo:", style_h2))
    pesos_data = [
        [Paragraph("<b>Fator Comportamental</b>", style_body_bold), Paragraph("<b>Coeficiente (&beta;)</b>", style_body_bold), Paragraph("<b>Interpretação para o Professor</b>", style_body_bold)],
        [Paragraph("<code>ansiedade_sem_celular</code>", style_code), Paragraph("<b>+0.5124</b>", style_body), Paragraph("Maior impacto relativo (+0.51). A abstinência digital é o principal gatilho de quebra de foco.", style_body)],
        [Paragraph("<code>dificuldade_concentracao</code>", style_code), Paragraph("+0.3756", style_body), Paragraph("Segundo fator mais relevante. Vulnerabilidade atencional prévia do estudante.", style_body)],
        [Paragraph("<code>busca_dopamina_scroll</code>", style_code), Paragraph("+0.3541", style_body), Paragraph("Uso impulsivo e involuntário das redes. Condiciona o cérebro a buscar estímulos rápidos.", style_body)],
        [Paragraph("<code>problemas_sono</code>", style_code), Paragraph("+0.1621", style_body), Paragraph("Cansaço diurno reduz o controle inibitório do córtex pré-frontal.", style_body)],
        [Paragraph("<code>horas_redes_sociais</code>", style_code), Paragraph("+0.1460", style_body), Paragraph("Efeito cumulativo do tempo diário em telas.", style_body)],
        [Paragraph("<b>Intercepto basal (&beta;0)</b>", style_body_bold), Paragraph("<b>-4.8891</b>", style_body), Paragraph("Ponto de partida neutro. Garante que, com hábitos saudáveis, o risco basal fique abaixo de 10%.", style_body)]
    ]
    t_pesos = Table(pesos_data, colWidths=[140, 95, 252])
    t_pesos.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_pesos)

    # ==================================================================
    # SEÇÃO 4: EXPERIMENTOS CIENTÍFICOS - experimentos.py
    # ==================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Experimentos Controlados (<code>experimentos.py</code>)", style_h1))
    story.append(Paragraph(
        "Corresponde ao <b>Passo 5 (Mude Uma Coisa de Cada Vez)</b> e <b>Passo 6 (Explique o Resultado)</b>. "
        "Para provar que a IA não aprendeu correlações espúrias, aplicou-se o princípio científico <i>ceteris paribus</i>: "
        "congelamos 4 fatores no valor médio da amostra e variamos apenas 1 ao longo de uma escala contínua.",
        style_body
    ))

    exp_data = [
        [Paragraph("<b>Experimento</b>", style_body_bold), Paragraph("<b>Variável Testada</b>", style_body_bold), Paragraph("<b>Fatores Congelados</b>", style_body_bold), Paragraph("<b>Comportamento Observado</b>", style_body_bold)],
        [Paragraph("<b>Exp. 1</b>", style_body_bold), Paragraph("<code>horas_redes</code> (0h a 7h)", style_code), Paragraph("Dopamina=3, Ansiedade=2, Sono=2, Foco=2", style_body), Paragraph("Curva ascendente e contínua provando que tempo de tela eleva o risco de dispersão.", style_body)],
        [Paragraph("<b>Exp. 2</b>", style_body_bold), Paragraph("<code>busca_dopamina</code> (1 a 5)", style_code), Paragraph("Horas=2.5h, Ansiedade=2, Sono=2, Foco=2", style_body), Paragraph("O risco dobra ao sair do uso consciente (1) para o scroll involuntário (5).", style_body)],
        [Paragraph("<b>Exp. 3</b>", style_body_bold), Paragraph("<code>ansiedade_celular</code> (1 a 5)", style_code), Paragraph("Horas=2.5h, Dopamina=3, Sono=2, Foco=2", style_body), Paragraph("Curva mais íngreme: passar de 1 para 5 eleva a probabilidade para mais de 70%.", style_body)]
    ]
    t_exp = Table(exp_data, colWidths=[55, 120, 140, 172])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#059669")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_exp)

    # ==================================================================
    # SEÇÃO 5: A APLICAÇÃO WEB HÍBRIDA - app.py
    # ==================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. A Aplicação Web e Chatbot (<code>app.py</code>)", style_h1))
    story.append(Paragraph(
        "O arquivo <code>app.py</code> é o núcleo da entrega interativa. Ele foi desenvolvido intencionalmente usando apenas a "
        "biblioteca padrão do Python (<code>http.server</code>, <code>socketserver</code>, <code>urllib.request</code>), <b>sem depender de Flask, Django ou Streamlit</b>. "
        "Isso garante que o professor possa clonar o repositório e rodar o projeto em qualquer máquina com um único comando: <code>python app.py</code>.",
        style_body
    ))

    story.append(Paragraph("5.1. Backend Python: Endpoints da API", style_h2))
    story.append(Paragraph(
        "A classe <code>FocusHandler(http.server.SimpleHTTPRequestHandler)</code> responde a três rotas principais:",
        style_body
    ))

    endpoints_data = [
        [Paragraph("<b>Rota</b>", style_body_bold), Paragraph("<b>Método</b>", style_body_bold), Paragraph("<b>Funcionamento Interno</b>", style_body_bold)],
        [Paragraph("<code>/</code>", style_code), Paragraph("GET", style_body), Paragraph("Entrega a página HTML completa com estilos e scripts embutidos.", style_body)],
        [Paragraph("<code>/api/predict</code>", style_code), Paragraph("POST", style_body), Paragraph("Recebe JSON com as 5 variáveis, converte para DataFrame e executa <code>modelo.predict_proba()</code> no modelo .joblib.", style_body)],
        [Paragraph("<code>/api/chat</code>", style_code), Paragraph("POST", style_body), Paragraph("Recebe mensagem e contexto do usuário. Tenta a Google Gemini API; se falhar ou não houver chave, aciona o fallback offline.", style_body)]
    ]
    t_end = Table(endpoints_data, colWidths=[80, 55, 352])
    t_end.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_end)

    story.append(Spacer(1, 10))
    story.append(Paragraph("5.2. Como o Chatbot se Conecta com o Machine Learning", style_h2))
    story.append(Paragraph(
        "A conexão entre a IA Preditiva e a IA Generativa ocorre via <b>Injeção de Contexto (Context Injection)</b>. "
        "Quando o usuário conversa com o Coach Zen, o backend monta dinamicamente o <code>system_prompt</code> injetando "
        "as probabilidades exatas calculadas pela Regressão Logística:",
        style_body
    ))

    code_prompt = """# Trecho de injeção de contexto em app.py
system_prompt = f\"\"\"Você é o Coach Zen de Neurociência e Foco.
O usuário passou pelo modelo preditivo com as seguintes métricas:
- Risco de Procrastinação: {ctx.get('probabilidade')}% ({ctx.get('status')})
- Tempo em Telas: {ctx.get('horas_redes_sociais')} h/dia
- Inquietação sem Celular: {ctx.get('ansiedade_sem_celular')}/5 (Maior peso no modelo: +0.51)

Instruções para o Coach:
- Seja extremamente direto, conciso e ágil nas respostas (máximo 2 a 3 frases).
- Foque 100% na ação física imediata para romper a inércia (protocolo 90s, regra dos 5 min).
- O objetivo é fazer o usuário fechar o chat e voltar a produzir imediatamente.\"\"\""""
    t_pr = Table([[Paragraph(code_prompt.replace("\n", "<br/>").replace(" ", "&nbsp;"), style_code)]], colWidths=[487])
    t_pr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_code_bg),
        ('BOX', (0,0), (-1,-1), 1, c_code_border),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_pr)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Modelo Utilizado:</b> <code>gemini-flash-lite-latest</code> (Google Gemini API). O modelo foi configurado com "
        "<code>maxOutputTokens: 250</code> e <code>temperature: 0.5</code> para respostas ultrarrápidas e sem alucinações prolixas.",
        style_body
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("5.3. Frontend: Pomodoro Configurável e Contabilização Automática", style_h2))
    story.append(Paragraph(
        "O frontend foi projetado em estética <i>Dark Mode Minimalista</i> (inspirada no Linear e Raycast) para evitar distrações. "
        "Ele contém recursos de ponta do mercado de produtividade:",
        style_body
    ))

    pomo_items = [
        "<b>Ciclos Pomodoro Completos:</b> Alternância fluida entre Foco (padrão 25m), Pausa Curta (5m) e Pausa Longa (15m a cada 4 rounds).",
        "<b>Totalmente Configurável:</b> Modal discreto para ajustar a minutagem de foco e pausas a qualquer momento, persistido em <code>localStorage</code>.",
        "<b>Contabilização Automática no Relatório:</b> Ao finalizar o timer de foco (ou clicar em 'Concluir Bloco'), os minutos são somados imediatamente na base de estudo do dia.",
        "<b>Áudio Zen Nativo:</b> Ruído de chuva sintetizado matematicamente via <i>Web Audio API</i> (sem carregar arquivos de áudio externos) e sino harmônico relaxante ao término do bloco.",
        "<b>Streaks & Analytics:</b> Contador de dias consecutivos de estudo e treino com chamas (&bull; 3+ dias) e grade de 14 dias estilo <i>GitHub contributions</i>."
    ]
    for p_item in pomo_items:
        story.append(Paragraph(f"• {p_item}", style_body))

    # ==================================================================
    # SEÇÃO 6: GUIA DE RESPOSTAS PARA A APRESENTAÇÃO COM PROF. WAGNER
    # ==================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Simulado: O que Responder para o Professor Wagner", style_h1))
    story.append(Paragraph(
        "Aqui estão as perguntas mais prováveis que o professor Wagner poderá fazer durante a arguição, "
        "acompanhadas da resposta técnica ideal:",
        style_body
    ))

    qa_list = [
        ("1. 'Por que você escolheu Regressão Logística e não uma Rede Neural Profunda?'",
         "Professor, o Passo 2 exige justificar a técnica e o Passo 6 exige explicabilidade. "
         "Uma rede neural seria uma 'caixa-preta' com pesos difíceis de interpretar. A Regressão Logística oferece transparência total: "
         "podemos demonstrar matematicamente o coeficiente de cada fator comportamental (como os +0.5124 da ansiedade sem celular) "
         "e mapear o risco contínuo através da curva Sigmoide sem risco de overfitting."),

        ("2. 'Essa base de 481 registros é sintética (inventada) ou real?'",
         "A base é 100% real. Ela foi obtida do repositório de pesquisa 'Social Media & Mental Health Survey' (smmh_raw.csv). "
         "Nós tratamos os dados, mapeamos as 5 variáveis comportamentais no script e mantivemos o arquivo bruto no repositório "
         "para garantir reprodutibilidade científica completa."),

        ("3. 'Por que a acurácia no teste foi de 68.04% e não 95%?'",
         "Em dados comportamentais humanos reais, obter 95% de acurácia com apenas 481 amostras seria um forte indício de overfitting "
         "(o algoritmo apenas memorizando ruído). 68% com equilíbrio de 70% na precisão e revocação demonstra que o modelo aprendeu "
         "a tendência geral da população e generaliza bem para novos estudantes nunca vistos durante o treino."),

        ("4. 'O que você fez no Passo 5 (Mudar uma coisa por vez)?'",
         "Aplicamos o método ceteris paribus no script experimentos.py. Congelamos 4 variáveis no valor médio e variamos apenas 1. "
         "Com isso, comprovamos que aumentar o tempo de tela eleva o risco de forma monotônica, e que a inquietação sem celular possui "
         "a inclinação mais íngreme de todo o modelo, gerando os gráficos de evidência científica."),

        ("5. 'Como a aplicação web funciona se a máquina estiver sem internet?'",
         "O sistema possui arquitetura resiliente. A inferência preditiva de risco roda localmente via modelo serializado .joblib "
         "(e a sigmoide está também espelhada no JavaScript para atualização em tempo real). Para o chatbot, se não houver internet "
         "ou chave de API, o motor ativa um fallback offline inteligente baseado em regras de neurociência, sem quebrar a aplicação."),

        ("6. 'Como as horas de estudo são contabilizadas na aplicação?'",
         "As horas são sincronizadas diretamente através do Pomodoro integrado. Cada bloco de foco concluído pelo estudante "
         "é automaticamente gravado no histórico de atividades diárias (localStorage), atualizando em tempo real o streak consecutivo, "
         "as horas totais e o gráfico semanal comparativo com os treinos físicos.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", style_h2))
        t_box = Table([[Paragraph(a, style_callout)]], colWidths=[487])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('LINELEFT', (0,0), (-1,-1), 3, c_accent),
            ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(t_box)
        story.append(Spacer(1, 6))

    # ==================================================================
    # SEÇÃO FINAL: COMANDOS PARA EXECUTAR NA APRESENTAÇÃO
    # ==================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Resumo Rápido de Comandos para a Apresentação", style_h1))
    cmd_data = [
        [Paragraph("<b>Comando</b>", style_body_bold), Paragraph("<b>Resultado que será exibido</b>", style_body_bold)],
        [Paragraph("<code>python treinar_modelo.py</code>", style_code), Paragraph("Exibe acurácia (68.04%), matriz de confusão e tabela de coeficientes beta.", style_body)],
        [Paragraph("<code>python experimentos.py</code>", style_code), Paragraph("Gera novamente os 5 gráficos científicos de validação na pasta <code>graficos/</code>.", style_body)],
        [Paragraph("<code>python app.py</code>", style_code), Paragraph("Abre o servidor em <code>http://localhost:5000</code> com os 6 módulos interativos.", style_body)],
        [Paragraph("<code>python assistente_foco.py</code>", style_code), Paragraph("Demonstração interativa de inferência direta no console do terminal.", style_body)]
    ]
    t_cmd = Table(cmd_data, colWidths=[170, 317])
    t_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF gerado com sucesso em: {os.path.abspath(caminho_saida)}")

if __name__ == '__main__':
    gerar_pdf()
