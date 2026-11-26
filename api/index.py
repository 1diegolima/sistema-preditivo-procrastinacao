import os
import sys
import json
import math
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

# Adiciona diretório raiz para importar HTML_PAGE do app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from app import HTML_PAGE
except Exception as e:
    HTML_PAGE = f"<h1>Erro ao carregar HTML: {e}</h1>"

# Coeficientes da Regressão Logística Real
B0 = -4.889100598331315
B_HORAS = 0.14603957
B_DOPAMINA = 0.35412845
B_ANSIEDADE = 0.51240182
B_SONO = 0.16213791
B_CONCENTRACAO = 0.37560214

def sigmoid(z):
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        url_path = self.path.split('?')[0]
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if url_path == '/api/predict':
            try:
                data = json.loads(post_data.decode('utf-8'))
                h = float(data.get('horas_redes_sociais', 2.0))
                d = int(data.get('busca_dopamina_scroll', 2))
                a = int(data.get('ansiedade_sem_celular', 2))
                s = int(data.get('problemas_sono', 2))
                c = int(data.get('dificuldade_concentracao', 2))

                z = B0 + (B_HORAS * h) + (B_DOPAMINA * d) + (B_ANSIEDADE * a) + (B_SONO * s) + (B_CONCENTRACAO * c)
                prob = sigmoid(z) * 100
                classe = 1 if prob >= 50.0 else 0

                resposta = {
                    'probabilidade': round(prob, 1),
                    'classe': classe,
                    'status': 'ALTO_RISCO' if classe == 1 else 'FOCO_PRESERVADO'
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resposta).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({'erro': str(e)}).encode('utf-8'))

        elif url_path == '/api/chat':
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_msg = data.get('message', '').strip()
                ctx = data.get('context', {})
                api_key = data.get('api_key', '').strip() or os.environ.get('GEMINI_API_KEY', '')

                reply = ""
                source = "offline"

                # 1. Tentativa de chamada à Google Gemini API
                if api_key:
                    try:
                        system_prompt = f"""Você é o Coach Zen de Neurociência e Foco do 'Sistema Preditivo de Foco'.
O usuário passou pelo modelo preditivo de Machine Learning com as seguintes métricas atuais:
- Risco de Procrastinação: {ctx.get('probabilidade', 50)}% ({ctx.get('status', 'MODERADO')})
- Tempo em Telas: {ctx.get('horas_redes_sociais', 2.0)} horas/dia
- Inquietação sem Celular (Abstinência): {ctx.get('ansiedade_sem_celular', 2)}/5 (Fator de maior peso no modelo: +0.51)
- Scroll sem Propósito (Dopamina): {ctx.get('busca_dopamina_scroll', 2)}/5
- Problemas de Sono: {ctx.get('problemas_sono', 2)}/5
- Dificuldade prévia de foco: {ctx.get('dificuldade_concentracao', 2)}/5

🚨 DIRETRIZ SUPREMA E CHOQUE DE REALIDADE (INVIOLÁVEL):
Sua ÚNICA E EXCLUSIVA função neste sistema é manter o usuário FOCADO em suas tarefas e combater a procrastinação.
1. SE O USUÁRIO PERGUNTAR QUALQUER COISA FORA DO ESCOPO DE FOCO/ESTUDO/PRODUTIVIDADE:
   (Exemplos: receitas de bolo/culinária, piadas, amenidades, fofocas, entretenimento, curiosidades aleatórias ou papo furado que claramente é uma fuga de atenção):
   ⛔ NÃO RESPONDA À PERGUNTA! NÃO DÊ DICAS DO ASSUNTO ALHEIO!
   Corte imediatamente e aplique um CHOQUE DE REALIDADE firme, direto e cirúrgico no usuário, apontando que ele está usando o chat como uma forma disfarçada de procrastinar e fugir da tarefa real.
   Exemplo de resposta obrigatória:
   "🛑 **Pare agora.** Você abriu este app para combater a procrastinação, mas está me perguntando sobre coisas aleatórias. Isso é o seu cérebro criando uma distração elaborada para fugir do trabalho real que você precisa fazer. Feche este chat agora mesmo, vá para a aba **Pomodoro** e inicie seu bloco de estudo imediatamente."
2. SE A PERGUNTA FOR LEGÍTIMA SOBRE FOCO, MATÉRIA DE ESTUDO, HÁBITOS OU COMO VENCER A INÉRCIA:
   - Responda em português brasileiro de forma acolhedora, porém extremamente direta e concisa (máximo 2 a 3 frases curtas).
   - Foque 100% na ação física imediata (regra dos 5 min, protocolo dos 90s, fricção visual, celular em outro cômodo).
   - O objetivo é fazer o usuário fechar o chat e ir agir imediatamente."""

                        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
                        gemini_payload = {
                            "system_instruction": {"parts": [{"text": system_prompt}]},
                            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 250}
                        }

                        req = urllib.request.Request(
                            gemini_url,
                            data=json.dumps(gemini_payload).encode('utf-8'),
                            headers={'Content-Type': 'application/json'}
                        )
                        with urllib.request.urlopen(req, timeout=12) as res:
                            res_json = json.loads(res.read().decode('utf-8'))
                            raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
                            import re
                            formatted = raw_text
                            formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted)
                            formatted = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted)
                            reply = formatted.replace('\n', '<br>')
                            source = "gemini-flash-lite-latest"
                    except Exception as gemini_err:
                        print(f"[!] Erro Gemini API na Vercel: {gemini_err}. Usando fallback.")
                        reply = ""

                # 2. Fallback Inteligente Offline
                if not reply:
                    lower = user_msg.lower()
                    prob = ctx.get('probabilidade', 50)
                    distracoes = ['bolo', 'receita', 'futebol', 'jogo', 'piada', 'fofoca', 'filme', 'série', 'musica', 'música', 'clima', 'tempo']
                    if any(w in lower for w in distracoes):
                        reply = "🛑 <strong>Pare agora.</strong> Você abriu este app para combater a procrastinação, mas está usando o chat para se distrair. Isso é o seu cérebro sabotando o seu foco.<br><br>👉 Feche esta janela agora, abra a aba <strong>Pomodoro</strong> e inicie seu bloco de foco imediatamente."
                    elif 'feed' in lower or 'sos' in lower or 'celular' in lower or 'tiktok' in lower or 'instagram' in lower:
                        reply = "🛑 <strong>PARE por 90 segundos!</strong><br><br>A neurociência comprova que a onda de urgência de dopamina decai se você não alimentar o estímulo nos primeiros 90 segundos.<br>1. Não encoste no aparelho.<br>2. Faça 2 respirações lentas pelo nariz.<br>3. Combine consigo mesmo: <em>'Vou fazer só 5 minutos da tarefa'</em>. O atrito está apenas no início!"
                    elif 'sono' in lower or 'dormir' in lower or 'cansaço' in lower:
                        reply = f"Com sono ruim, seu córtex pré-frontal tem menos energia inibitória (seu risco calculado é de {prob}%).<br><br>👉 <strong>Estratégia Imediata:</strong> Não tente blocos de 2 horas. Use <strong>Micro-Pomodoros de 15 minutos</strong> na aba Pomodoro. Conclua apenas 1 prioridade essencial hoje."
                    elif 'ansiedade' in lower or 'abstinência' in lower:
                        reply = "O modelo de IA apontou que <strong>a inquietação sem celular tem o maior peso (+0.51)</strong> na sua propensão a dispersar.<br><br>👉 <strong>Tática Imediata:</strong> Coloque a tela do celular em <em>Escala de Cinza</em> nas configurações e guarde-o fora da mesa. Sem cores vibrantes, o cérebro encerra a busca compulsiva por estímulos."
                    else:
                        reply = f"Seu risco preditivo atual é de <strong>{prob}%</strong>.<br><br>Minha recomendação direta: vá para a aba <strong>Pomodoro</strong>, defina 1 única micro-meta e execute 1 bloco de foco com a Chuva Zen ligada. A dopamina virá da sensação de progresso real."
                    source = "offline-zen"

                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({'reply': reply, 'source': source}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({'erro': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
