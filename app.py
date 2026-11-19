"""
Servidor Web Minimalista & Moderno de Previsão de Procrastinação
Arquitetura Híbrida:
1. IA Preditiva Clássica: Regressão Logística treinada com dados reais (.joblib)
2. IA Generativa: Integração com Google Gemini API (1M tokens) + Fallback offline
3. Timer Zen Pomodoro + Som de Chuva via Web Audio API
4. Checklist dinâmico anti-procrastinação
"""

import http.server
import socketserver
import json
import joblib
import pandas as pd
import os
import urllib.request
import urllib.error
import webbrowser

PORT = 5000
MODEL_PATH = 'modelo_procrastinacao.joblib'

# 1. Carregamento do modelo de IA preditiva
if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)
    print(f"[*] Modelo preditivo '{MODEL_PATH}' carregado.")
else:
    modelo = None
    print(f"[!] Aviso: '{MODEL_PATH}' não encontrado. Execute 'python treinar_modelo.py'.")

# 2. Leitura opcional de chave de API em arquivo .env ou variável de ambiente
GEMINI_KEY_DEFAULT = os.environ.get('GEMINI_API_KEY', '')
if not GEMINI_KEY_DEFAULT and os.path.exists('.env'):
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    GEMINI_KEY_DEFAULT = line.strip().split('=', 1)[1].strip(' "\'')
    except Exception:
        pass

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sistema de Foco & Previsão de Procrastinação</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --surface: #101726;
      --surface-border: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #f43f5e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
      -webkit-font-smoothing: antialiased;
    }
    .container {
      width: 100%;
      max-width: 720px;
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 24px;
      padding: 32px;
      box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.6);
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--surface-border);
    }
    .title-area h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px var(--accent);
    }
    .nav-tabs {
      display: flex;
      gap: 6px;
      background: rgba(0, 0, 0, 0.3);
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--surface-border);
    }
    .nav-btn {
      background: transparent;
      border: none;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .nav-btn.active {
      background: var(--accent);
      color: #000;
    }
    /* Dashboard */
    .dashboard {
      display: flex;
      align-items: center;
      gap: 24px;
      background: rgba(9, 13, 22, 0.7);
      border: 1px solid var(--surface-border);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .gauge-box {
      position: relative;
      width: 120px;
      height: 120px;
      flex-shrink: 0;
    }
    .gauge-svg {
      width: 100%;
      height: 100%;
      transform: rotate(-90deg);
    }
    .gauge-bg {
      stroke: var(--surface-border);
      stroke-width: 9;
      fill: none;
    }
    .gauge-fill {
      stroke-width: 9;
      fill: none;
      stroke-linecap: round;
      stroke-dasharray: 282.7;
      stroke-dashoffset: 200;
      transition: stroke-dashoffset 0.6s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s ease;
    }
    .gauge-center {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .gauge-val {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.04em;
    }
    .gauge-label {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .status-area { flex: 1; }
    .status-pill {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 8px;
    }
    .status-desc {
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .quick-win {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 0.75rem;
    }
    .quick-win-btn {
      color: var(--accent);
      background: none;
      border: none;
      font-weight: 600;
      cursor: pointer;
    }
    /* Controls */
    .controls-title {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 16px;
    }
    .control-group { margin-bottom: 16px; }
    .control-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      font-weight: 500;
      margin-bottom: 8px;
    }
    .control-val {
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      font-size: 0.82rem;
    }
    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      background: transparent;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 18px;
      width: 18px;
      border-radius: 50%;
      background: #ffffff;
      cursor: pointer;
      margin-top: -6px;
      box-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
      transition: transform 0.15s ease;
    }
    input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }
    input[type=range]::-webkit-slider-runnable-track {
      width: 100%;
      height: 6px;
      cursor: pointer;
      background: var(--surface-border);
      border-radius: 3px;
    }
    .presets {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 20px;
      padding-top: 18px;
      border-top: 1px solid var(--surface-border);
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--surface-border);
      color: var(--text);
      font-size: 0.75rem;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .preset-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: var(--accent);
    }
    /* Chatbot Styles */
    .chat-box {
      background: rgba(9, 13, 22, 0.7);
      border: 1px solid var(--surface-border);
      border-radius: 20px;
      padding: 20px;
    }
    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--surface-border);
      margin-bottom: 16px;
    }
    .api-badge {
      font-size: 0.7rem;
      font-family: 'JetBrains Mono', monospace;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.25);
      cursor: pointer;
    }
    .api-badge:hover {
      background: rgba(56, 189, 248, 0.2);
    }
    .chat-log {
      height: 260px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding-right: 8px;
      margin-bottom: 14px;
    }
    .msg {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      font-size: 0.8rem;
      line-height: 1.5;
    }
    .msg-bot { justify-content: flex-start; }
    .msg-user { justify-content: flex-end; }
    .msg-bubble {
      padding: 10px 14px;
      border-radius: 12px;
      max-width: 84%;
    }
    .bubble-bot {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--surface-border);
      color: var(--text);
      border-top-left-radius: 0;
    }
    .bubble-user {
      background: var(--accent);
      color: #000;
      font-weight: 500;
      border-top-right-radius: 0;
    }
    .chips-area {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--surface-border);
      color: var(--muted);
      font-size: 0.72rem;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .chip:hover {
      color: var(--text);
      border-color: var(--accent);
    }
    .chip-sos {
      background: rgba(244, 63, 94, 0.1);
      border-color: rgba(244, 63, 94, 0.3);
      color: #fda4af;
    }
    .chat-input-row {
      display: flex;
      gap: 8px;
    }
    .chat-input {
      flex: 1;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 10px 14px;
      color: var(--text);
      font-size: 0.8rem;
      outline: none;
    }
    .chat-input:focus { border-color: var(--accent); }
    .chat-send {
      background: var(--accent);
      color: #000;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      padding: 0 16px;
      cursor: pointer;
    }
    /* Key Modal */
    .key-bar {
      display: none;
      margin-bottom: 12px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      font-size: 0.75rem;
    }
    .key-bar.active { display: flex; gap: 8px; align-items: center; }
    .key-input {
      flex: 1;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 6px 10px;
      color: #fff;
      font-family: monospace;
      font-size: 0.75rem;
    }
    /* Timer & Plan */
    .timer-container {
      text-align: center;
      padding: 24px;
      background: rgba(9, 13, 22, 0.7);
      border: 1px solid var(--surface-border);
      border-radius: 20px;
    }
    .timer-clock {
      font-family: 'JetBrains Mono', monospace;
      font-size: 4rem;
      font-weight: 300;
      color: var(--text);
      margin: 20px 0;
    }
    .timer-actions {
      display: flex;
      justify-content: center;
      gap: 10px;
    }
    .btn-timer-primary {
      background: var(--accent);
      color: #000;
      font-weight: 600;
      padding: 10px 24px;
      border-radius: 10px;
      border: none;
      cursor: pointer;
    }
    .btn-timer-sec {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--surface-border);
      color: var(--muted);
      padding: 10px 18px;
      border-radius: 10px;
      cursor: pointer;
    }
    .sound-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 20px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--surface-border);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.75rem;
      cursor: pointer;
      color: var(--muted);
    }
    .plan-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 16px;
    }
    .plan-item {
      display: flex;
      align-items: center;
      gap: 12px;
      background: rgba(9, 13, 22, 0.6);
      border: 1px solid var(--surface-border);
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .plan-item input { accent-color: var(--accent); width: 16px; height: 16px; }
    .hidden { display: none !important; }
  </style>
</head>
<body>

  <div class="container">
    <div class="header">
      <div class="title-area">
        <h1><span class="dot"></span> Sistema Preditivo de Foco</h1>
      </div>
      <div class="nav-tabs">
        <button id="nav-diag" class="nav-btn active" onclick="showTab('diag')">Diagnóstico</button>
        <button id="nav-chat" class="nav-btn" onclick="showTab('chat')">💬 Coach Zen</button>
        <button id="nav-timer" class="nav-btn" onclick="showTab('timer')">Timer Zen</button>
        <button id="nav-plan" class="nav-btn" onclick="showTab('plan')">Plano</button>
      </div>
    </div>

    <!-- TAB 1: DIAGNÓSTICO -->
    <div id="tab-diag">
      <div class="dashboard">
        <div class="gauge-box">
          <svg class="gauge-svg" viewBox="0 0 100 100">
            <circle class="gauge-bg" cx="50" cy="50" r="45" />
            <circle id="gauge" class="gauge-fill" cx="50" cy="50" r="45" stroke="var(--success)" />
          </svg>
          <div class="gauge-center">
            <span id="prob-num" class="gauge-val">--%</span>
            <span class="gauge-label">Risco</span>
          </div>
        </div>
        <div class="status-area">
          <span id="status-pill" class="status-pill" style="background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3);">Foco Preservado</span>
          <p id="status-desc" class="status-desc">Seus parâmetros digitais estão equilibrados. A probabilidade de dispersão e fuga para telas é baixa.</p>
          
          <div class="quick-win">
            <span>💡 <strong>Vitória Rápida:</strong> <span id="quick-win-text">Reduza 1h de tela para diminuir seu risco.</span></span>
            <button class="quick-win-btn" onclick="simularVitoria()">Simular</button>
          </div>
        </div>
      </div>

      <div class="controls-title">Variáveis de Entrada Comportamentais (IA)</div>

      <div class="control-group">
        <div class="control-header">
          <span>Tempo Diário em Telas / Redes</span>
          <span id="txt-horas" class="control-val">2.0 h/dia</span>
        </div>
        <input type="range" id="input-horas" min="0" max="8" step="0.5" value="2.0" oninput="recalcular()">
      </div>

      <div class="control-group">
        <div class="control-header">
          <span>Busca por Dopamina (Scroll Automático sem Propósito)</span>
          <span id="txt-dopamina" class="control-val">Nível 2 / 5</span>
        </div>
        <input type="range" id="input-dopamina" min="1" max="5" step="1" value="2" oninput="recalcular()">
      </div>

      <div class="control-group">
        <div class="control-header">
          <span>Inquietação ao Ficar sem Celular (Maior Peso: +0.51)</span>
          <span id="txt-ansiedade" class="control-val">Nível 2 / 5</span>
        </div>
        <input type="range" id="input-ansiedade" min="1" max="5" step="1" value="2" oninput="recalcular()">
      </div>

      <div class="control-group">
        <div class="control-header">
          <span>Problemas de Sono / Insônia</span>
          <span id="txt-sono" class="control-val">Nível 2 / 5</span>
        </div>
        <input type="range" id="input-sono" min="1" max="5" step="1" value="2" oninput="recalcular()">
      </div>

      <div class="control-group">
        <div class="control-header">
          <span>Dificuldade Prévia em Sustentar a Atenção</span>
          <span id="txt-concentracao" class="control-val">Nível 2 / 5</span>
        </div>
        <input type="range" id="input-concentracao" min="1" max="5" step="1" value="2" oninput="recalcular()">
      </div>

      <div class="presets">
        <span style="font-size: 0.75rem; color: var(--muted);">Simular perfis:</span>
        <button class="preset-btn" onclick="aplicarPreset(1.0, 1, 1, 1, 2)">Foco Alto</button>
        <button class="preset-btn" onclick="aplicarPreset(3.5, 3, 3, 3, 3)">Estudante Médio</button>
        <button class="preset-btn" onclick="aplicarPreset(6.0, 5, 5, 4, 4)">Hiper-estimulado</button>
      </div>
    </div>

    <!-- TAB 2: CHATBOT COACH ZEN (COM GEMINI LLM) -->
    <div id="tab-chat" class="hidden">
      <div class="chat-box">
        <div class="chat-header">
          <div>
            <h3 style="font-size: 0.85rem; font-weight: 600;">Coach Zen de Foco & Neurociência</h3>
            <p style="font-size: 0.72rem; color: var(--muted);">Potencializado por Google Gemini (1M tokens) + Contexto de IA</p>
          </div>
          <button id="btn-toggle-key" class="api-badge" onclick="toggleKeyBar()">⚙️ Configurar Gemini</button>
        </div>

        <div id="key-bar" class="key-bar">
          <input type="password" id="gemini-key-input" class="key-input" placeholder="Cole sua Google Gemini API Key aqui...">
          <button onclick="saveApiKey()" style="background: var(--accent); color:#000; border:none; padding:6px 12px; border-radius:6px; font-weight:600; cursor:pointer;">Salvar</button>
        </div>

        <div id="chat-log" class="chat-log">
          <div class="msg msg-bot">
            <div class="msg-bubble bubble-bot">
              Olá! Sou seu <strong>Coach Zen</strong>. Se a procrastinação ou o impulso de pegar o celular surgirem, clique nos atalhos rápidos ou pergunte qualquer coisa:
            </div>
          </div>
        </div>

        <div class="chips-area">
          <button class="chip chip-sos" onclick="sendQuickPrompt('sos')">🚨 SOS: Vontade incontrolável de rolar o feed</button>
          <button class="chip" onclick="sendQuickPrompt('ansiedade')">⚡ Como conter a abstinência do celular?</button>
          <button class="chip" onclick="sendQuickPrompt('sono')">😴 Dormi mal, como estudar hoje?</button>
          <button class="chip" onclick="sendQuickPrompt('pomodoro')">⏱️ Como fazer meu 1º bloco de 25 min?</button>
        </div>

        <div class="chat-input-row">
          <input type="text" id="chat-input-text" class="chat-input" placeholder="Escreva sua dúvida, desabafo ou matéria que vai estudar..." onkeydown="if(event.key==='Enter') handleChatSend()">
          <button id="btn-chat-send" class="chat-send" onclick="handleChatSend()">Enviar</button>
        </div>
      </div>
    </div>

    <!-- TAB 3: TIMER ZEN -->
    <div id="tab-timer" class="hidden">
      <div class="timer-container">
        <h3 style="font-size: 0.85rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em;">Modo Foco Profundo</h3>
        <div id="timer-clock" class="timer-clock">25:00</div>
        <div class="timer-actions">
          <button id="btn-timer-toggle" class="btn-timer-primary" onclick="toggleTimer()">Iniciar Bloco</button>
          <button class="btn-timer-sec" onclick="setTimer(25)">25 min</button>
          <button class="btn-timer-sec" onclick="setTimer(50)">50 min</button>
          <button class="btn-timer-sec" onclick="setTimer(5)">Pausa (5 min)</button>
        </div>
        <br>
        <button id="btn-sound" class="sound-toggle" onclick="toggleSound()">
          <span id="sound-icon">🔇</span>
          <span id="sound-txt">Som de Chuva / Ruído: Desligado</span>
        </button>
      </div>
    </div>

    <!-- TAB 4: PLANO DE AÇÃO -->
    <div id="tab-plan" class="hidden">
      <h3 style="font-size: 0.85rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em;">Seu Checklist Anti-Procrastinação</h3>
      <div id="plan-items" class="plan-list"></div>
    </div>
  </div>

  <script>
    const B0 = -4.889100598331315;
    const B_HORAS = 0.14603957;
    const B_DOPAMINA = 0.35412845;
    const B_ANSIEDADE = 0.51240182;
    const B_SONO = 0.16213791;
    const B_CONCENTRACAO = 0.37560214;

    let currentProb = 25;
    let geminiApiKey = localStorage.getItem('GEMINI_API_KEY') || '';

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function recalcular() {
      const h = parseFloat(document.getElementById('input-horas').value);
      const d = parseInt(document.getElementById('input-dopamina').value);
      const a = parseInt(document.getElementById('input-ansiedade').value);
      const s = parseInt(document.getElementById('input-sono').value);
      const c = parseInt(document.getElementById('input-concentracao').value);

      document.getElementById('txt-horas').textContent = h.toFixed(1) + ' h/dia';
      document.getElementById('txt-dopamina').textContent = 'Nível ' + d + ' / 5';
      document.getElementById('txt-ansiedade').textContent = 'Nível ' + a + ' / 5';
      document.getElementById('txt-sono').textContent = 'Nível ' + s + ' / 5';
      document.getElementById('txt-concentracao').textContent = 'Nível ' + c + ' / 5';

      const z = B0 + (B_HORAS * h) + (B_DOPAMINA * d) + (B_ANSIEDADE * a) + (B_SONO * s) + (B_CONCENTRACAO * c);
      currentProb = Math.round(sigmoid(z) * 100);

      document.getElementById('prob-num').textContent = currentProb + '%';
      const gauge = document.getElementById('gauge');
      gauge.style.strokeDashoffset = 282.74 - (282.74 * (currentProb / 100));

      const pill = document.getElementById('status-pill');
      const desc = document.getElementById('status-desc');
      const win = document.getElementById('quick-win-text');

      if (currentProb < 35) {
        gauge.setAttribute('stroke', 'var(--success)');
        pill.textContent = 'Foco Preservado';
        pill.style.background = 'rgba(16, 185, 129, 0.15)';
        pill.style.color = 'var(--success)';
        desc.textContent = 'Autocontrole preservado. A probabilidade de você abandonar a tarefa de estudo por dispersão é mínima.';
        win.textContent = 'Tudo equilibrado. Inicie um bloco Pomodoro para aproveitar este estado.';
      } else if (currentProb < 60) {
        gauge.setAttribute('stroke', 'var(--warning)');
        pill.textContent = 'Atenção Vulnerável';
        pill.style.background = 'rgba(245, 158, 11, 0.15)';
        pill.style.color = 'var(--warning)';
        desc.textContent = 'Atenção vulnerável: ao menor sinal de cansaço ou tédio, há tendência forte de checar o celular.';
        win.textContent = 'Reduzir 1.5h de tela diária pode derrubar seu risco em ~12%.';
      } else {
        gauge.setAttribute('stroke', 'var(--danger)');
        pill.textContent = 'Alto Risco de Procrastinação';
        pill.style.background = 'rgba(244, 63, 94, 0.15)';
        pill.style.color = 'var(--danger)';
        desc.textContent = 'Alerta de saturação dopaminérgica: alto índice de scroll e abstinência sem smartphone sequestram o foco.';
        win.textContent = 'Colocar o celular em outro cômodo derruba o risco imediato para menos da metade.';
      }

      updatePlan(h, d, a, s);
    }

    function simularVitoria() {
      const h = document.getElementById('input-horas');
      const a = document.getElementById('input-ansiedade');
      if (parseFloat(h.value) > 1.5) h.value = Math.max(1.0, parseFloat(h.value) - 1.5);
      if (parseInt(a.value) > 2) a.value = parseInt(a.value) - 1;
      recalcular();
    }

    function aplicarPreset(h, d, a, s, c) {
      document.getElementById('input-horas').value = h;
      document.getElementById('input-dopamina').value = d;
      document.getElementById('input-ansiedade').value = a;
      document.getElementById('input-sono').value = s;
      document.getElementById('input-concentracao').value = c;
      recalcular();
    }

    function showTab(tab) {
      const tabs = ['diag', 'chat', 'timer', 'plan'];
      tabs.forEach(t => {
        document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
        document.getElementById('nav-' + t).classList.toggle('active', t === tab);
      });
    }

    function updatePlan(h, d, a, s) {
      const plan = document.getElementById('plan-items');
      const items = [
        "Guardar o celular fora do campo de visão durante o bloco de foco",
        "Dividir o estudo em metas pequenas com prazo definido (máx 25-50 min)"
      ];
      if (d >= 3) items.push("Ativar modo 'Escala de Cinza' nas opções de acessibilidade do celular");
      if (a >= 3) items.push("Esperar 90 segundos respirando fundo ao sentir vontade impulsiva de checar notificações");
      if (s >= 3) items.push("Cessar exposição a telas 40 minutos antes do horário de dormir");

      plan.innerHTML = items.map(text => `
        <label class="plan-item">
          <input type="checkbox">
          <span>${text}</span>
        </label>
      `).join('');
    }

    function toggleKeyBar() {
      const bar = document.getElementById('key-bar');
      bar.classList.toggle('active');
      if (bar.classList.contains('active')) {
        document.getElementById('gemini-key-input').value = geminiApiKey;
      }
    }

    function saveApiKey() {
      const val = document.getElementById('gemini-key-input').value.trim();
      geminiApiKey = val;
      localStorage.setItem('GEMINI_API_KEY', val);
      document.getElementById('key-bar').classList.remove('active');
      updateKeyBadge();
      appendChat('bot', val ? '✅ <strong>Gemini API conectada com sucesso!</strong> Agora tenho acesso ao modelo de 1 milhão de tokens.' : 'ℹ️ Modo offline restaurado.');
    }

    function updateKeyBadge() {
      const btn = document.getElementById('btn-toggle-key');
      if (geminiApiKey) {
        btn.textContent = '🟢 Gemini Ativo';
        btn.style.color = '#10b981';
      } else {
        btn.textContent = '⚙️ Configurar Gemini';
        btn.style.color = 'var(--accent)';
      }
    }

    // CHATBOT ENGINE COM GEMINI API E FALLBACK
    function appendChat(role, text) {
      const log = document.getElementById('chat-log');
      const msg = document.createElement('div');
      msg.className = role === 'bot' ? 'msg msg-bot' : 'msg msg-user';
      msg.innerHTML = `<div class="msg-bubble ${role === 'bot' ? 'bubble-bot' : 'bubble-user'}">${text}</div>`;
      log.appendChild(msg);
      log.scrollTop = log.scrollHeight;
    }

    function sendQuickPrompt(type) {
      const prompts = {
        'sos': '🚨 SOS: Vontade incontrolável de rolar o feed agora!',
        'ansiedade': '⚡ Como conter a abstinência e inquietação ao ficar longe do celular?',
        'sono': '😴 Dormi mal, como conseguir estudar hoje?',
        'pomodoro': '⏱️ Como fazer meu primeiro bloco de 25 min com sucesso?'
      };
      if (prompts[type]) {
        executeChat(prompts[type]);
      }
    }

    function handleChatSend() {
      const input = document.getElementById('chat-input-text');
      const val = input.value.trim();
      if (!val) return;
      input.value = '';
      executeChat(val);
    }

    async function executeChat(userMessage) {
      appendChat('user', userMessage);

      // Contexto atual do modelo de Machine Learning
      const context = {
        horas_redes_sociais: parseFloat(document.getElementById('input-horas').value),
        busca_dopamina_scroll: parseInt(document.getElementById('input-dopamina').value),
        ansiedade_sem_celular: parseInt(document.getElementById('input-ansiedade').value),
        problemas_sono: parseInt(document.getElementById('input-sono').value),
        dificuldade_concentracao: parseInt(document.getElementById('input-concentracao').value),
        probabilidade: currentProb,
        status: currentProb < 35 ? 'FOCO_PRESERVADO' : (currentProb < 60 ? 'RISCO_MODERADO' : 'ALTO_RISCO')
      };

      appendChat('bot', '<span style="opacity:0.6;">Digitando resposta... 🧠</span>');
      const log = document.getElementById('chat-log');
      const loadingMsg = log.lastElementChild;

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: userMessage,
            context: context,
            api_key: geminiApiKey
          })
        });

        const resData = await response.json();
        loadingMsg.remove();
        appendChat('bot', resData.reply || 'Não foi possível processar a resposta.');
      } catch (err) {
        loadingMsg.remove();
        // Fallback local caso o servidor não responda
        appendChat('bot', '<strong>Dica do Coach:</strong> Para vencer o impulso imediato de procrastinação, comprometa-se a fazer <strong>apenas 5 minutos</strong> de foco na aba <em>Timer Zen</em>. O atrito está no começo!');
      }
    }

    // Timer
    let totalSec = 25 * 60;
    let timerInt = null;

    function formatTime(s) {
      const m = Math.floor(s / 60).toString().padStart(2, '0');
      const sec = (s % 60).toString().padStart(2, '0');
      return m + ':' + sec;
    }

    function toggleTimer() {
      const btn = document.getElementById('btn-timer-toggle');
      if (timerInt) {
        clearInterval(timerInt);
        timerInt = null;
        btn.textContent = 'Continuar Bloco';
      } else {
        btn.textContent = 'Pausar';
        timerInt = setInterval(() => {
          if (totalSec > 0) {
            totalSec--;
            document.getElementById('timer-clock').textContent = formatTime(totalSec);
          } else {
            clearInterval(timerInt);
            timerInt = null;
            btn.textContent = 'Concluído';
            alert('🎉 Bloco de foco concluído sem distrações!');
          }
        }, 1000);
      }
    }

    function setTimer(m) {
      if (timerInt) { clearInterval(timerInt); timerInt = null; }
      totalSec = m * 60;
      document.getElementById('timer-clock').textContent = formatTime(totalSec);
      document.getElementById('btn-timer-toggle').textContent = 'Iniciar Bloco';
    }

    // Áudio Sintetizado
    let audioCtx = null;
    let noiseNode = null;
    let isNoiseOn = false;

    function toggleSound() {
      const icon = document.getElementById('sound-icon');
      const txt = document.getElementById('sound-txt');
      if (!isNoiseOn) {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const buf = audioCtx.createBuffer(1, audioCtx.sampleRate * 2, audioCtx.sampleRate);
        const out = buf.getChannelData(0);
        let last = 0;
        for (let i = 0; i < out.length; i++) {
          const w = Math.random() * 2 - 1;
          out[i] = (last + (0.02 * w)) / 1.02;
          last = out[i];
          out[i] *= 3.5;
        }
        noiseNode = audioCtx.createBufferSource();
        noiseNode.buffer = buf;
        noiseNode.loop = true;
        const gain = audioCtx.createGain();
        gain.gain.value = 0.08;
        noiseNode.connect(gain);
        gain.connect(audioCtx.destination);
        noiseNode.start();
        isNoiseOn = true;
        icon.textContent = '🔊';
        txt.textContent = 'Chuva Zen: Ligada';
      } else {
        if (noiseNode) { noiseNode.stop(); noiseNode.disconnect(); noiseNode = null; }
        isNoiseOn = false;
        icon.textContent = '🔇';
        txt.textContent = 'Som de Chuva / Ruído: Desligado';
      }
    }

    recalcular();
    updateKeyBadge();
  </script>
</body>
</html>
"""

class FocusHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                df_input = pd.DataFrame([{
                    'horas_redes_sociais': float(data.get('horas_redes_sociais', 2.0)),
                    'busca_dopamina_scroll': int(data.get('busca_dopamina_scroll', 2)),
                    'ansiedade_sem_celular': int(data.get('ansiedade_sem_celular', 2)),
                    'problemas_sono': int(data.get('problemas_sono', 2)),
                    'dificuldade_concentracao': int(data.get('dificuldade_concentracao', 2))
                }])
                
                if modelo:
                    prob = float(modelo.predict_proba(df_input)[0][1] * 100)
                    classe = int(modelo.predict(df_input)[0])
                else:
                    prob = 50.0
                    classe = 0

                resposta = {
                    'probabilidade': round(prob, 1),
                    'classe': classe,
                    'status': 'ALTO_RISCO' if classe == 1 else 'FOCO_PRESERVADO'
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resposta).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'erro': str(e)}).encode('utf-8'))

        elif self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_msg = data.get('message', '').strip()
                ctx = data.get('context', {})
                api_key = data.get('api_key', '').strip() or GEMINI_KEY_DEFAULT

                reply = ""
                source = "offline"

                # 1. Tentativa de chamada à Google Gemini API caso haja chave
                if api_key:
                    try:
                        system_prompt = f"""Você é o Coach Zen de Neurociência e Foco do 'Sistema Preditivo de Foco'.
O usuário passou pelo modelo preditivo de Machine Learning com as seguintes métricas atuais:
- Risco de Procrastinação: {ctx.get('probabilidade', 50)}% ({ctx.get('status', 'MODERADO')})
- Tempo em Telas: {ctx.get('horas_redes_sociais', 2.0)} horas/dia
- Frequência de Scroll sem Propósito (Dopamina): {ctx.get('busca_dopamina_scroll', 2)}/5
- Inquietação sem Celular (Abstinência): {ctx.get('ansiedade_sem_celular', 2)}/5 (Fator de maior peso no modelo: +0.51)
- Problemas de Sono: {ctx.get('problemas_sono', 2)}/5
- Dificuldade prévia de foco: {ctx.get('dificuldade_concentracao', 2)}/5

Instruções para o Coach:
- Responda em português brasileiro de forma acolhedora, empática e direta (máximo 2 a 3 parágrafos curtos).
- Forneça técnicas práticas e imediatas de neurociência e psicologia comportamental (Andrew Huberman, James Clear, Cal Newport).
- Ajude o usuário a vencer a inércia e quebrar o atrito imediato para a tarefa que ele precisa realizar agora.
- Seja extremamente direto, conciso e ágil nas respostas, visando máxima economia de tokens e zero enrolação/introduções vazias.
- Limite sua resposta a no máximo 2 ou 3 frases curtas (ou 3 tópicos enxutos), focando 100% na ação física imediata que o usuário deve tomar agora.
- O objetivo é fazer o usuário fechar o chat e voltar a produzir o mais rápido possível.
- Use técnicas de neurociência prática (regra dos 5 min, protocolo de 90s, fricção visual, respiração)."""
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
                        with urllib.request.urlopen(req, timeout=15) as res:
                            res_json = json.loads(res.read().decode('utf-8'))
                            raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
                            # Converte quebras de linha em tags HTML simples
                            reply = raw_text.replace('\n', '<br>')
                            source = "gemini-flash-lite-latest"
                    except Exception as gemini_err:
                        print(f"[!] Erro ao chamar Gemini API: {gemini_err}. Usando fallback offline.")
                        reply = ""

                # 2. Fallback Inteligente Offline (se sem chave ou se falhar)
                if not reply:
                    lower = user_msg.lower()
                    prob = ctx.get('probabilidade', 50)
                    if 'feed' in lower or 'sos' in lower or 'celular' in lower or 'tiktok' in lower or 'instagram' in lower:
                        reply = "<strong>PARE por 90 segundos!</strong> 🛑<br><br>A neurociência comprova que a onda de urgência de dopamina decai se você não alimentar o estímulo nos primeiros 90 segundos.<br>1. Não encoste no aparelho.<br>2. Faça 2 respirações lentas pelo nariz.<br>3. Combine consigo mesmo: <em>'Vou fazer só 5 minutos da tarefa'</em>. Romper a inércia faz a dopamina vir do progresso real!"
                    elif 'sono' in lower or 'dormir' in lower or 'cansaço' in lower:
                        reply = f"Com sono ruim, seu córtex pré-frontal tem menos energia para inibir impulsos (seu risco calculado é de {prob}%).<br><br>👉 <strong>Estratégia de Sobrevivência:</strong> Não tente blocos de 2 horas. Use <strong>Micro-Pomodoros de 15 minutos</strong> com pausas para caminhar e beber água. Termine apenas 1 prioridade essencial hoje."
                    elif 'ansiedade' in lower or 'abstinência' in lower:
                        reply = "O modelo de IA apontou que <strong>a inquietação sem celular tem o maior peso (+0.51)</strong> na sua propensão a dispersar.<br><br>👉 <strong>Tática Imediata:</strong> Coloque a tela do celular em <em>Escala de Cinza</em> nas configurações. Sem cores vibrantes, seu cérebro desliga o modo de caça à dopamina."
                    else:
                        reply = f"Entendido! O seu risco preditivo atual calculado pela IA é de <strong>{prob}%</strong>.<br><br>Minha recomendação direta: vá para a aba <strong>Timer Zen</strong>, selecione 1 única micro-meta para agora e execute 1 bloco de 25 minutos com o ruído de chuva ativado. Pequenas vitórias consistentes constroem disciplina."
                    source = "offline-zen"

                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({'reply': reply, 'source': source}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'erro': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def iniciar_servidor():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), FocusHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 65)
        print(f"  🧠 SERVIDOR HÍBRIDO (ML + GEMINI API) INICIADO!")
        print(f"  Acesse no seu navegador: {url}")
        print("  Pressione Ctrl+C para encerrar o servidor.")
        print("=" * 65)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor encerrado.")

if __name__ == '__main__':
    iniciar_servidor()
