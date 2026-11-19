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
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #080c14;
      --card-bg: rgba(15, 23, 42, 0.65);
      --card-border: rgba(255, 255, 255, 0.07);
      --card-hover: rgba(255, 255, 255, 0.1);
      --text: #f8fafc;
      --text-secondary: #94a3b8;
      --text-tertiary: #64748b;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.15);
      --warning: #f59e0b;
      --danger: #f43f5e;
      --danger-glow: rgba(244, 63, 94, 0.15);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.05) 0%, transparent 60%);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 32px 16px;
      -webkit-font-smoothing: antialiased;
    }
    .container {
      width: 100%;
      max-width: 740px;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 32px;
      box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.03);
    }
    
    /* Top Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--card-border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 10px var(--accent);
    }
    .brand h1 {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }
    .brand-tag {
      font-size: 0.65rem;
      font-family: 'JetBrains Mono', monospace;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* Minimalist Segmented Navbar */
    .nav-bar {
      display: flex;
      gap: 4px;
      background: rgba(0, 0, 0, 0.35);
      padding: 4px;
      border-radius: var(--radius-md);
      border: 1px solid var(--card-border);
      margin-bottom: 24px;
      overflow-x: auto;
    }
    .nav-item {
      flex: 1;
      min-width: 90px;
      text-align: center;
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      white-space: nowrap;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .nav-item:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }
    .nav-item.active {
      background: #1e293b;
      color: #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Common Card / Panel */
    .card {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-tertiary);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Toast Notification */
    .toast-notice {
      position: fixed;
      top: 24px;
      left: 50%;
      transform: translateX(-50%) translateY(-20px);
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 20px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 500;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      opacity: 0;
      pointer-events: none;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 9999;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .toast-notice.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }

    /* ========================================================
       TAB 1: DIAGNÓSTICO (PREVISÃO ML)
       ======================================================== */
    .dashboard-grid {
      display: flex;
      align-items: center;
      gap: 24px;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 24px;
      margin-bottom: 24px;
    }
    .gauge-wrapper {
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
      stroke: rgba(255, 255, 255, 0.06);
      stroke-width: 8;
      fill: none;
    }
    .gauge-fill {
      stroke-width: 8;
      fill: none;
      stroke-linecap: round;
      stroke-dasharray: 282.7;
      stroke-dashoffset: 200;
      transition: stroke-dashoffset 0.6s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.4s ease;
    }
    .gauge-center-content {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .gauge-number {
      font-size: 1.7rem;
      font-weight: 700;
      letter-spacing: -0.04em;
      font-family: 'JetBrains Mono', monospace;
    }
    .gauge-sub {
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-tertiary);
    }
    .dash-info { flex: 1; }
    .dash-badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 999px;
      margin-bottom: 8px;
    }
    .dash-text {
      font-size: 0.8rem;
      color: var(--text-secondary);
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .control-item { margin-bottom: 14px; }
    .control-header-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 500;
      margin-bottom: 6px;
    }
    .control-val-badge {
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      font-size: 0.75rem;
    }
    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      background: transparent;
      outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 16px;
      width: 16px;
      border-radius: 50%;
      background: #ffffff;
      cursor: pointer;
      margin-top: -5px;
      box-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
      transition: transform 0.15s ease;
    }
    input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.15); }
    input[type=range]::-webkit-slider-runnable-track {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
    }
    .presets-row {
      display: flex;
      gap: 8px;
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--card-border);
    }
    .btn-preset {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      color: var(--text-secondary);
      font-size: 0.72rem;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-preset:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
    }

    /* ========================================================
       TAB 2: POMODORO MINIMALISTA & CONFIGURÁVEL
       ======================================================== */
    .pomodoro-card {
      text-align: center;
      padding: 32px 24px;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      position: relative;
    }
    /* Mode Selector Pills */
    .pomo-modes {
      display: inline-flex;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      padding: 3px;
      border-radius: 999px;
      margin-bottom: 24px;
    }
    .pomo-mode-btn {
      background: transparent;
      border: none;
      color: var(--text-tertiary);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 6px 16px;
      border-radius: 999px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .pomo-mode-btn.active {
      background: #1e293b;
      color: #fff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    /* Clock Display */
    .pomo-clock {
      font-family: 'JetBrains Mono', monospace;
      font-size: 5rem;
      font-weight: 300;
      letter-spacing: -0.04em;
      line-height: 1;
      color: #fff;
      margin: 10px 0 16px 0;
    }
    /* Progress Bar */
    .pomo-progress-track {
      width: 100%;
      max-width: 320px;
      height: 4px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      margin: 0 auto 20px auto;
      overflow: hidden;
    }
    .pomo-progress-bar {
      height: 100%;
      background: var(--accent);
      width: 0%;
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    /* Task Input */
    .pomo-task-input {
      background: transparent;
      border: none;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.15);
      color: var(--text);
      font-size: 0.85rem;
      text-align: center;
      padding: 6px 12px;
      width: 80%;
      max-width: 380px;
      margin-bottom: 26px;
      outline: none;
      transition: border-color 0.2s ease;
    }
    .pomo-task-input:focus {
      border-bottom-color: var(--accent);
    }
    .pomo-task-input::placeholder {
      color: var(--text-tertiary);
      font-style: italic;
    }
    /* Round Indicator */
    .pomo-rounds {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-bottom: 24px;
      font-size: 0.72rem;
      color: var(--text-tertiary);
    }
    .round-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.1);
    }
    .round-dot.done { background: var(--accent); }
    .round-dot.current { box-shadow: 0 0 8px var(--accent); background: #fff; }
    /* Primary Controls */
    .pomo-controls {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }
    .btn-pomo-main {
      background: var(--accent);
      color: #040810;
      font-weight: 700;
      font-size: 0.9rem;
      padding: 12px 36px;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      letter-spacing: 0.03em;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      box-shadow: 0 4px 16px var(--accent-glow);
    }
    .btn-pomo-main:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(56, 189, 248, 0.3);
    }
    .btn-pomo-sec {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-secondary);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 11px 18px;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-pomo-sec:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
    }
    /* Secondary Actions (Sound & Settings) */
    .pomo-footer-actions {
      display: flex;
      justify-content: center;
      gap: 12px;
      font-size: 0.72rem;
      color: var(--text-tertiary);
      margin-top: 10px;
    }
    .pomo-action-link {
      background: none;
      border: none;
      color: var(--text-tertiary);
      font-size: 0.72rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      border-radius: 6px;
      transition: all 0.2s ease;
    }
    .pomo-action-link:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
    }
    .pomo-action-link.active {
      color: var(--accent);
    }

    /* Modal de Configuração do Pomodoro */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      z-index: 10000;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }
    .modal-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }
    .modal-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 24px;
      width: 100%;
      max-width: 420px;
      box-shadow: 0 25px 60px rgba(0, 0, 0, 0.8);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .modal-header h3 {
      font-size: 0.95rem;
      font-weight: 700;
      color: #fff;
    }
    .btn-close-modal {
      background: none;
      border: none;
      color: var(--text-tertiary);
      font-size: 1.1rem;
      cursor: pointer;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }
    .settings-field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .settings-field label {
      font-size: 0.68rem;
      color: var(--text-tertiary);
      font-weight: 500;
    }
    .settings-field input {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      text-align: center;
      outline: none;
    }
    .settings-field input:focus { border-color: var(--accent); }
    .btn-save-settings {
      width: 100%;
      background: var(--accent);
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: 10px;
      padding: 10px;
      cursor: pointer;
      font-size: 0.8rem;
    }

    /* ========================================================
       TAB 3: COACH ZEN (CHATBOT GEMINI)
       ======================================================== */
    .chat-container {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 20px;
    }
    .chat-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 14px;
    }
    .chat-log-box {
      height: 260px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding-right: 6px;
      margin-bottom: 14px;
    }
    .chat-bubble-row {
      display: flex;
      gap: 8px;
      font-size: 0.8rem;
      line-height: 1.5;
    }
    .chat-bubble-row.bot { justify-content: flex-start; }
    .chat-bubble-row.user { justify-content: flex-end; }
    .chat-bubble {
      padding: 10px 14px;
      border-radius: 14px;
      max-width: 82%;
    }
    .chat-bubble.bot {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: var(--text);
      border-top-left-radius: 2px;
    }
    .chat-bubble.user {
      background: var(--accent);
      color: #000;
      font-weight: 500;
      border-top-right-radius: 2px;
    }
    .chat-quick-chips {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .quick-chip {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      color: var(--text-secondary);
      font-size: 0.7rem;
      padding: 4px 10px;
      border-radius: 999px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s ease;
    }
    .quick-chip:hover {
      border-color: var(--accent);
      color: #fff;
    }
    .chat-input-bar {
      display: flex;
      gap: 8px;
    }
    .chat-text-input {
      flex: 1;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 14px;
      color: #fff;
      font-size: 0.8rem;
      outline: none;
    }
    .chat-text-input:focus { border-color: var(--accent); }
    .chat-send-btn {
      background: var(--accent);
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: 10px;
      padding: 0 16px;
      cursor: pointer;
      font-size: 0.8rem;
    }

    /* ========================================================
       TAB 4: ATIVIDADES (TREINO & REGISTRO)
       ======================================================== */
    .activity-grid {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .pomo-sync-banner {
      background: rgba(56, 189, 248, 0.06);
      border: 1px dashed rgba(56, 189, 248, 0.25);
      border-radius: var(--radius-md);
      padding: 14px 16px;
      font-size: 0.78rem;
      color: var(--accent);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .form-inline {
      display: flex;
      gap: 8px;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    .form-field {
      display: flex;
      flex-direction: column;
      gap: 4px;
      flex: 1;
      min-width: 110px;
    }
    .form-field label {
      font-size: 0.68rem;
      color: var(--text-tertiary);
      font-weight: 600;
      text-transform: uppercase;
    }
    .form-field input, .form-field select {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      color: #fff;
      font-size: 0.78rem;
      outline: none;
    }
    .form-field input:focus, .form-field select:focus {
      border-color: var(--accent);
    }
    .btn-action-primary {
      background: var(--accent);
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: var(--radius-sm);
      padding: 8px 16px;
      cursor: pointer;
      font-size: 0.78rem;
      height: 35px;
      white-space: nowrap;
    }
    .btn-action-success {
      background: var(--success);
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: var(--radius-sm);
      padding: 8px 16px;
      cursor: pointer;
      font-size: 0.78rem;
      height: 35px;
      white-space: nowrap;
    }
    .history-scroll {
      max-height: 160px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 10px;
    }
    .history-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      padding: 7px 10px;
      font-size: 0.75rem;
    }
    .history-date {
      color: var(--text-tertiary);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
    }
    .history-dur {
      font-weight: 700;
      color: var(--accent);
      font-family: 'JetBrains Mono', monospace;
    }
    .history-desc {
      color: var(--text-secondary);
      flex: 1;
      margin-left: 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* ========================================================
       TAB 5: RELATÓRIO ANALYTICS & STREAKS
       ======================================================== */
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 20px;
    }
    .metric-card {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 18px;
      text-align: center;
      transition: all 0.2s ease;
    }
    .metric-card:hover { border-color: rgba(255, 255, 255, 0.15); }
    .metric-icon { font-size: 1.4rem; margin-bottom: 6px; }
    .metric-value {
      font-size: 1.8rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.04em;
      color: #fff;
    }
    .metric-label {
      font-size: 0.68rem;
      color: var(--text-tertiary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 4px;
    }
    
    /* Streak Container */
    .streak-container {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 20px;
      margin-bottom: 20px;
    }
    .streak-line {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 10px 0;
    }
    .streak-line:not(:last-child) {
      border-bottom: 1px solid var(--card-border);
    }
    .streak-num {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--accent);
    }
    .streak-tiles {
      display: flex;
      gap: 4px;
      margin-top: 8px;
    }
    .streak-tile {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.06);
    }
    .streak-tile.active { background: var(--success); }
    .streak-tile.today { border: 1px solid var(--accent); }

    /* Weekly Chart */
    .chart-container {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 20px;
    }
    .chart-bars {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      height: 120px;
      padding-top: 10px;
    }
    .chart-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-end;
      height: 100%;
      gap: 4px;
    }
    .col-bars-stack {
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
    }
    .col-seg {
      width: 65%;
      border-radius: 3px;
      min-height: 0px;
      transition: height 0.3s ease;
    }
    .col-seg.study { background: var(--accent); }
    .col-seg.train { background: var(--success); }
    .col-label {
      font-size: 0.65rem;
      color: var(--text-tertiary);
    }
    .col-val {
      font-size: 0.6rem;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-secondary);
    }

    /* ========================================================
       TAB 6: PLANO DE AÇÃO
       ======================================================== */
    .checklist-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .check-item {
      display: flex;
      align-items: center;
      gap: 12px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .check-item:hover { border-color: rgba(255, 255, 255, 0.12); }
    .check-item input { accent-color: var(--accent); width: 16px; height: 16px; }

    .hidden { display: none !important; }
  </style>
</head>
<body>

  <!-- Toast Notification -->
  <div id="toast" class="toast-notice"></div>

  <div class="container">
    <!-- Header -->
    <div class="header">
      <div class="brand">
        <span class="brand-dot"></span>
        <h1>Sistema Preditivo & Foco</h1>
        <span class="brand-tag">v2.0 Minimal</span>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-bar">
      <button id="nav-diag" class="nav-item active" onclick="showTab('diag')">🎯 Diagnóstico</button>
      <button id="nav-pomo" class="nav-item" onclick="showTab('pomo')">⏱️ Pomodoro</button>
      <button id="nav-chat" class="nav-item" onclick="showTab('chat')">💬 Coach Zen</button>
      <button id="nav-ativ" class="nav-item" onclick="showTab('ativ')">🏋️ Atividades</button>
      <button id="nav-report" class="nav-item" onclick="showTab('report')">📊 Relatório</button>
      <button id="nav-plan" class="nav-item" onclick="showTab('plan')">📋 Plano</button>
    </div>

    <!-- ==========================================
         TAB 1: DIAGNÓSTICO
         ========================================== -->
    <div id="tab-diag">
      <div class="dashboard-grid">
        <div class="gauge-wrapper">
          <svg class="gauge-svg" viewBox="0 0 100 100">
            <circle class="gauge-bg" cx="50" cy="50" r="45" />
            <circle id="gauge" class="gauge-fill" cx="50" cy="50" r="45" stroke="var(--success)" />
          </svg>
          <div class="gauge-center-content">
            <span id="prob-num" class="gauge-number">25%</span>
            <span class="gauge-sub">Risco ML</span>
          </div>
        </div>
        <div class="dash-info">
          <span id="status-pill" class="dash-badge" style="background: rgba(16, 185, 129, 0.15); color: var(--success);">Foco Preservado</span>
          <p id="status-desc" class="dash-text">Seus hábitos atuais protegem sua atenção contra a procrastinação involuntária.</p>
          <button class="btn-preset" onclick="simularVitoria()" style="border-color: var(--accent); color: var(--accent);">⚡ Simular Redução de Telas (-1.5h)</button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Seus Hábitos Diários (Entradas do Modelo)</div>
        
        <div class="control-item">
          <div class="control-header-row">
            <span>Tempo em Redes Sociais</span>
            <span id="txt-horas" class="control-val-badge">2.0 h/dia</span>
          </div>
          <input type="range" id="input-horas" min="0.5" max="8.0" step="0.5" value="2.0" oninput="recalcular()">
        </div>

        <div class="control-item">
          <div class="control-header-row">
            <span>Scroll sem Propósito (Busca por Dopamina)</span>
            <span id="txt-dopamina" class="control-val-badge">Nível 2 / 5</span>
          </div>
          <input type="range" id="input-dopamina" min="1" max="5" step="1" value="2" oninput="recalcular()">
        </div>

        <div class="control-item">
          <div class="control-header-row">
            <span>Inquietação sem Celular (Abstinência)</span>
            <span id="txt-ansiedade" class="control-val-badge">Nível 2 / 5</span>
          </div>
          <input type="range" id="input-ansiedade" min="1" max="5" step="1" value="2" oninput="recalcular()">
        </div>

        <div class="control-item">
          <div class="control-header-row">
            <span>Problemas com Sono / Cansaço Diurno</span>
            <span id="txt-sono" class="control-val-badge">Nível 2 / 5</span>
          </div>
          <input type="range" id="input-sono" min="1" max="5" step="1" value="2" oninput="recalcular()">
        </div>

        <div class="control-item">
          <div class="control-header-row">
            <span>Dificuldade Prévia de Concentração</span>
            <span id="txt-concentracao" class="control-val-badge">Nível 2 / 5</span>
          </div>
          <input type="range" id="input-concentracao" min="1" max="5" step="1" value="2" oninput="recalcular()">
        </div>

        <div class="presets-row">
          <button class="btn-preset" onclick="aplicarPreset(1.5, 1, 1, 1, 1)">🟢 Estudante Zen</button>
          <button class="btn-preset" onclick="aplicarPreset(3.5, 3, 3, 3, 3)">🟡 Sobrecarga Média</button>
          <button class="btn-preset" onclick="aplicarPreset(6.0, 5, 5, 4, 4)">🔴 Saturação Dopamina</button>
        </div>
      </div>
    </div>

    <!-- ==========================================
         TAB 2: POMODORO MINIMALISTA & CONFIGURÁVEL
         ========================================== -->
    <div id="tab-pomo" class="hidden">
      <div class="pomodoro-card">
        <!-- Mode Pills -->
        <div class="pomo-modes">
          <button id="pomo-mode-focus" class="pomo-mode-btn active" onclick="setPomoMode('focus')">Foco</button>
          <button id="pomo-mode-short" class="pomo-mode-btn" onclick="setPomoMode('short')">Pausa Curta</button>
          <button id="pomo-mode-long" class="pomo-mode-btn" onclick="setPomoMode('long')">Pausa Longa</button>
        </div>

        <!-- Clock -->
        <div id="pomo-clock-display" class="pomo-clock">25:00</div>

        <!-- Progress Track -->
        <div class="pomo-progress-track">
          <div id="pomo-progress-bar" class="pomo-progress-bar"></div>
        </div>

        <!-- Task Input -->
        <div>
          <input type="text" id="pomo-task-name" class="pomo-task-input" placeholder="📌 No que você vai focar agora? (ex: Revisão de Algoritmos)" maxlength="50">
        </div>

        <!-- Rounds Indicator -->
        <div class="pomo-rounds">
          <span>Ciclo</span>
          <div id="round-dots-area" style="display:flex;gap:4px;">
            <div class="round-dot current"></div>
            <div class="round-dot"></div>
            <div class="round-dot"></div>
            <div class="round-dot"></div>
          </div>
          <span id="pomo-round-text">#1 de 4</span>
        </div>

        <!-- Main Controls -->
        <div class="pomo-controls">
          <button id="btn-pomo-main" class="btn-pomo-main" onclick="togglePomoTimer()">INICIAR</button>
          <button class="btn-pomo-sec" onclick="skipPomoCycle()" title="Avançar ciclo">Pular</button>
          <button class="btn-pomo-sec" onclick="finishCurrentBlockNow()" title="Concluir e contabilizar agora">Concluir Bloco</button>
        </div>

        <!-- Sub Actions -->
        <div class="pomo-footer-actions">
          <button id="btn-sound-toggle" class="pomo-action-link" onclick="toggleRainSound()">
            <span id="rain-icon">🌧️</span>
            <span id="rain-txt">Chuva Zen: Desligada</span>
          </button>
          <span>•</span>
          <button class="pomo-action-link" onclick="openPomoSettingsModal()">
            ⚙️ Ajustar Tempos
          </button>
        </div>
      </div>
    </div>

    <!-- ==========================================
         TAB 3: COACH ZEN (CHATBOT GEMINI)
         ========================================== -->
    <div id="tab-chat" class="hidden">
      <div class="chat-container">
        <div class="chat-top">
          <span style="font-size:0.75rem;font-weight:600;color:var(--text-secondary);">Coach de Atenção & Neurociência</span>
          <span id="gemini-status-badge" class="brand-tag" onclick="toggleKeyBar()" style="cursor:pointer;">🟢 Gemini Flash-Lite Ativo</span>
        </div>

        <!-- Key bar opcional -->
        <div id="key-bar" style="display:none;margin-bottom:12px;padding:8px;background:rgba(255,255,255,0.03);border-radius:8px;">
          <input type="password" id="gemini-key-input" placeholder="Cole sua Google Gemini API Key se desejar trocar..." style="width:100%;background:rgba(0,0,0,0.4);border:1px solid var(--card-border);padding:6px 10px;color:#fff;border-radius:6px;font-size:0.75rem;">
          <button onclick="salvarGeminiKeyManual()" class="btn-preset" style="margin-top:6px;">Salvar Chave</button>
        </div>

        <div id="chat-messages" class="chat-log-box">
          <div class="chat-bubble-row bot">
            <div class="chat-bubble bot">
              Olá! Sou seu <strong>Coach Zen</strong>. Estou sincronizado com o diagnóstico do seu modelo preditivo. Em que posso te ajudar a destravar agora?
            </div>
          </div>
        </div>

        <div class="chat-quick-chips">
          <button class="quick-chip" onclick="sendQuickPrompt('sos')">🚨 SOS: Vontade incontrolável de rolar o feed</button>
          <button class="quick-chip" onclick="sendQuickPrompt('ansiedade')">⚡ Inquietação sem o celular</button>
          <button class="quick-chip" onclick="sendQuickPrompt('sono')">😴 Dormi mal, como focar?</button>
        </div>

        <div class="chat-input-bar">
          <input type="text" id="chat-input-text" class="chat-text-input" placeholder="Pergunte ao Coach ou desabafe..." onkeydown="if(event.key==='Enter') handleChatSend()">
          <button class="chat-send-btn" onclick="handleChatSend()">Enviar</button>
        </div>
      </div>
    </div>

    <!-- ==========================================
         TAB 4: ATIVIDADES (TREINO & ESTUDO)
         ========================================== -->
    <div id="tab-ativ" class="hidden">
      <div class="activity-grid">
        <!-- Banner Pomodoro Sync -->
        <div class="pomo-sync-banner">
          <div>
            <strong>⏱️ Contabilização Automática:</strong> Suas sessões de estudo são somadas automaticamente ao usar a aba <strong>Pomodoro</strong>.
          </div>
          <button class="btn-preset" onclick="showTab('pomo')" style="background:var(--accent);color:#000;font-weight:700;">Ir para o Pomodoro →</button>
        </div>

        <!-- Registro de Atividade Física (Treinos) -->
        <div class="card">
          <div class="card-title">🏋️ Registrar Atividade Física</div>
          <div class="form-inline">
            <div class="form-field">
              <label>Data</label>
              <input type="date" id="train-date">
            </div>
            <div class="form-field">
              <label>Duração (min)</label>
              <input type="number" id="train-duration" min="5" max="360" value="45">
            </div>
            <div class="form-field">
              <label>Modalidade</label>
              <select id="train-type">
                <option value="Musculação">Musculação</option>
                <option value="Corrida">Corrida</option>
                <option value="Caminhada">Caminhada</option>
                <option value="Funcional / Calistenia">Funcional / Calistenia</option>
                <option value="Esporte">Esporte</option>
                <option value="Outro">Outro</option>
              </select>
            </div>
            <button class="btn-action-success" onclick="salvarTreinoManual()">+ Salvar Treino</button>
          </div>
          <div id="train-history-list" class="history-scroll"></div>
        </div>

        <!-- Registro Manual de Estudo (Avulso) -->
        <div class="card">
          <div class="card-title">📚 Registro Manual de Estudo (Opcional)</div>
          <div class="form-inline">
            <div class="form-field">
              <label>Data</label>
              <input type="date" id="study-date">
            </div>
            <div class="form-field">
              <label>Duração (min)</label>
              <input type="number" id="study-duration" min="5" max="600" value="30">
            </div>
            <div class="form-field" style="flex:2;">
              <label>Matéria / Assunto</label>
              <input type="text" id="study-note" placeholder="Ex: Exercícios de Álgebra">
            </div>
            <button class="btn-action-primary" onclick="salvarEstudoManual()">+ Registrar</button>
          </div>
          <div id="study-history-list" class="history-scroll"></div>
        </div>

        <div style="text-align:right;">
          <button onclick="limparTodoHistorico()" style="background:none;border:none;color:var(--danger);font-size:0.72rem;cursor:pointer;">🗑️ Limpar todo o histórico</button>
        </div>
      </div>
    </div>

    <!-- ==========================================
         TAB 5: RELATÓRIO ANALYTICS & STREAKS
         ========================================== -->
    <div id="tab-report" class="hidden">
      <!-- 4 Cards de Métricas -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-icon">📚</div>
          <div id="rpt-study-hours" class="metric-value">0h</div>
          <div class="metric-label">Horas de Estudo (Pomodoro)</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon">🏋️</div>
          <div id="rpt-train-days" class="metric-value">0</div>
          <div class="metric-label">Dias Treinados</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon">⏱️</div>
          <div id="rpt-study-sessions" class="metric-value">0</div>
          <div class="metric-label">Sessões de Foco</div>
        </div>
        <div class="metric-card">
          <div class="metric-icon">🔥</div>
          <div id="rpt-train-hours" class="metric-value">0h</div>
          <div class="metric-label">Horas de Treino</div>
        </div>
      </div>

      <!-- Streaks -->
      <div class="streak-container">
        <div class="card-title">🔥 Sequência Consecutiva (Streaks)</div>
        
        <div class="streak-line">
          <span style="font-size:1.3rem;">📚</span>
          <div style="flex:1;">
            <div style="font-size:0.8rem;font-weight:600;">Estudo & Foco</div>
            <div style="font-size:0.68rem;color:var(--text-tertiary);">Dias consecutivos com sessões concluídas</div>
            <div id="study-tiles" class="streak-tiles"></div>
          </div>
          <div style="display:flex;align-items:center;gap:4px;">
            <span id="rpt-study-streak" class="streak-num">0</span>
            <span id="rpt-study-fire"></span>
          </div>
        </div>

        <div class="streak-line">
          <span style="font-size:1.3rem;">🏋️</span>
          <div style="flex:1;">
            <div style="font-size:0.8rem;font-weight:600;">Atividade Física</div>
            <div style="font-size:0.68rem;color:var(--text-tertiary);">Dias consecutivos de treino registrado</div>
            <div id="train-tiles" class="streak-tiles"></div>
          </div>
          <div style="display:flex;align-items:center;gap:4px;">
            <span id="rpt-train-streak" class="streak-num">0</span>
            <span id="rpt-train-fire"></span>
          </div>
        </div>
      </div>

      <!-- Weekly Chart -->
      <div class="chart-container">
        <div class="card-title">
          <span>📊 Dedicação nos Últimos 7 Dias</span>
          <div style="display:flex;gap:12px;font-size:0.68rem;font-weight:400;text-transform:none;">
            <span><span style="display:inline-block;width:8px;height:8px;background:var(--accent);border-radius:2px;margin-right:4px;"></span>Estudo</span>
            <span><span style="display:inline-block;width:8px;height:8px;background:var(--success);border-radius:2px;margin-right:4px;"></span>Treino</span>
          </div>
        </div>
        <div id="weekly-bars-area" class="chart-bars"></div>
      </div>
    </div>

    <!-- ==========================================
         TAB 6: PLANO DE AÇÃO
         ========================================== -->
    <div id="tab-plan" class="hidden">
      <div class="card">
        <div class="card-title">Checklist Prático Anti-Dispersão</div>
        <div id="plan-items" class="checklist-container"></div>
      </div>
    </div>

  </div> <!-- /container -->

  <!-- Modal Configurações do Pomodoro -->
  <div id="modal-pomo-settings" class="modal-overlay" onclick="if(event.target===this) closePomoSettingsModal()">
    <div class="modal-box">
      <div class="modal-header">
        <h3>⚙️ Configuração do Pomodoro</h3>
        <button class="btn-close-modal" onclick="closePomoSettingsModal()">✕</button>
      </div>

      <div class="settings-grid">
        <div class="settings-field">
          <label>Foco (min)</label>
          <input type="number" id="cfg-pomo-focus" min="1" max="120" value="25">
        </div>
        <div class="settings-field">
          <label>Pausa Curta</label>
          <input type="number" id="cfg-pomo-short" min="1" max="60" value="5">
        </div>
        <div class="settings-field">
          <label>Pausa Longa</label>
          <input type="number" id="cfg-pomo-long" min="1" max="60" value="15">
        </div>
      </div>

      <div style="margin-bottom:18px;">
        <label style="font-size:0.75rem;color:var(--text-secondary);display:flex;align-items:center;gap:8px;cursor:pointer;">
          <input type="checkbox" id="cfg-pomo-sound" checked accent-color="var(--accent)">
          Tocar sinal sonoro ao finalizar bloco
        </label>
      </div>

      <button class="btn-save-settings" onclick="salvarConfigPomodoro()">Salvar & Aplicar</button>
    </div>
  </div>

  <script>
    /* ========================================================
       1. CÁLCULO PREDITIVO (MODELO REGRESSÃO LOGÍSTICA REAL)
       ======================================================== */
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

      if (currentProb < 35) {
        gauge.setAttribute('stroke', 'var(--success)');
        pill.textContent = 'Foco Preservado';
        pill.style.background = 'rgba(16, 185, 129, 0.15)';
        pill.style.color = 'var(--success)';
        desc.textContent = 'Autocontrole preservado. A probabilidade de você abandonar a tarefa de estudo por dispersão é mínima.';
      } else if (currentProb < 60) {
        gauge.setAttribute('stroke', 'var(--warning)');
        pill.textContent = 'Atenção Vulnerável';
        pill.style.background = 'rgba(245, 158, 11, 0.15)';
        pill.style.color = 'var(--warning)';
        desc.textContent = 'Atenção vulnerável: ao menor sinal de cansaço ou tédio, há tendência forte de checar o celular.';
      } else {
        gauge.setAttribute('stroke', 'var(--danger)');
        pill.textContent = 'Alto Risco de Procrastinação';
        pill.style.background = 'rgba(244, 63, 94, 0.15)';
        pill.style.color = 'var(--danger)';
        desc.textContent = 'Alerta de saturação dopaminérgica: alto índice de scroll e abstinência sem smartphone sequestram o foco.';
      }

      updatePlan(h, d, a, s);
    }

    function simularVitoria() {
      const h = document.getElementById('input-horas');
      const a = document.getElementById('input-ansiedade');
      if (parseFloat(h.value) > 1.5) h.value = Math.max(1.0, parseFloat(h.value) - 1.5);
      if (parseInt(a.value) > 2) a.value = parseInt(a.value) - 1;
      recalcular();
      showToast('⚡ Simulação aplicada: -1.5h de tela e menor abstinência!');
    }

    function aplicarPreset(h, d, a, s, c) {
      document.getElementById('input-horas').value = h;
      document.getElementById('input-dopamina').value = d;
      document.getElementById('input-ansiedade').value = a;
      document.getElementById('input-sono').value = s;
      document.getElementById('input-concentracao').value = c;
      recalcular();
    }

    function updatePlan(h, d, a, s) {
      const plan = document.getElementById('plan-items');
      const items = [
        "Guardar o celular fora do campo de visão durante o bloco de foco",
        "Dividir a matéria em blocos Pomodoro configurados de foco"
      ];
      if (d >= 3) items.push("Ativar modo 'Escala de Cinza' nas opções de acessibilidade do celular");
      if (a >= 3) items.push("Esperar 90 segundos respirando fundo ao sentir vontade impulsiva de checar notificações");
      if (s >= 3) items.push("Cessar exposição a telas 40 minutos antes do horário de dormir");

      plan.innerHTML = items.map(text => `
        <label class="check-item">
          <input type="checkbox">
          <span>${text}</span>
        </label>
      `).join('');
    }

    /* ========================================================
       2. NAVEGAÇÃO DE ABAS
       ======================================================== */
    function showTab(tab) {
      const tabs = ['diag', 'pomo', 'chat', 'ativ', 'report', 'plan'];
      tabs.forEach(t => {
        const el = document.getElementById('tab-' + t);
        const nav = document.getElementById('nav-' + t);
        if (el) el.classList.toggle('hidden', t !== tab);
        if (nav) nav.classList.toggle('active', t === tab);
      });
      if (tab === 'ativ') renderActivityHistories();
      if (tab === 'report') renderReportAnalytics();
    }

    /* ========================================================
       3. SISTEMA DE TOAST
       ======================================================== */
    let toastTimeout = null;
    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.innerHTML = msg;
      toast.classList.add('show');
      if (toastTimeout) clearTimeout(toastTimeout);
      toastTimeout = setTimeout(() => { toast.classList.remove('show'); }, 3000);
    }

    /* ========================================================
       4. POMODORO ENGINE COMPLETA & CONFIGURÁVEL
       ======================================================== */
    let pomoSettings = {
      focusMin: 25,
      shortMin: 5,
      longMin: 15,
      sound: true
    };
    try {
      const savedCfg = localStorage.getItem('pomodoroConfig');
      if (savedCfg) pomoSettings = Object.assign(pomoSettings, JSON.parse(savedCfg));
    } catch(e){}

    let currentPomoMode = 'focus'; // 'focus', 'short', 'long'
    let pomoTimerInt = null;
    let pomoTotalSeconds = pomoSettings.focusMin * 60;
    let pomoInitialSeconds = pomoTotalSeconds;
    let pomoRound = 1;

    function initPomoDisplay() {
      pomoTotalSeconds = getPomoDuration(currentPomoMode) * 60;
      pomoInitialSeconds = pomoTotalSeconds;
      updatePomoClockDisplay();
      updatePomoProgressBar();
      updatePomoRoundUI();
    }

    function getPomoDuration(mode) {
      if (mode === 'focus') return pomoSettings.focusMin;
      if (mode === 'short') return pomoSettings.shortMin;
      if (mode === 'long') return pomoSettings.longMin;
      return 25;
    }

    function formatClock(sec) {
      const m = Math.floor(sec / 60).toString().padStart(2, '0');
      const s = (sec % 60).toString().padStart(2, '0');
      return m + ':' + s;
    }

    function updatePomoClockDisplay() {
      document.getElementById('pomo-clock-display').textContent = formatClock(pomoTotalSeconds);
      document.title = `${formatClock(pomoTotalSeconds)} - ${currentPomoMode === 'focus' ? 'Foco' : 'Pausa'}`;
    }

    function updatePomoProgressBar() {
      const bar = document.getElementById('pomo-progress-bar');
      const pct = pomoInitialSeconds > 0 ? ((pomoInitialSeconds - pomoTotalSeconds) / pomoInitialSeconds) * 100 : 0;
      bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
      if (currentPomoMode === 'focus') {
        bar.style.background = 'var(--accent)';
      } else {
        bar.style.background = 'var(--success)';
      }
    }

    function updatePomoRoundUI() {
      const dotsArea = document.getElementById('round-dots-area');
      const txt = document.getElementById('pomo-round-text');
      txt.textContent = `#${pomoRound} de 4`;
      let dotsHtml = '';
      for (let i = 1; i <= 4; i++) {
        if (i < pomoRound) dotsHtml += '<div class="round-dot done"></div>';
        else if (i === pomoRound) dotsHtml += '<div class="round-dot current"></div>';
        else dotsHtml += '<div class="round-dot"></div>';
      }
      dotsArea.innerHTML = dotsHtml;
    }

    function setPomoMode(mode) {
      if (pomoTimerInt) {
        clearInterval(pomoTimerInt);
        pomoTimerInt = null;
        document.getElementById('btn-pomo-main').textContent = 'INICIAR';
      }
      currentPomoMode = mode;
      ['focus', 'short', 'long'].forEach(m => {
        const btn = document.getElementById('pomo-mode-' + m);
        if (btn) btn.classList.toggle('active', m === mode);
      });
      initPomoDisplay();
    }

    function togglePomoTimer() {
      const btn = document.getElementById('btn-pomo-main');
      if (pomoTimerInt) {
        clearInterval(pomoTimerInt);
        pomoTimerInt = null;
        btn.textContent = 'CONTINUAR';
      } else {
        btn.textContent = 'PAUSAR';
        pomoTimerInt = setInterval(() => {
          if (pomoTotalSeconds > 0) {
            pomoTotalSeconds--;
            updatePomoClockDisplay();
            updatePomoProgressBar();
          } else {
            clearInterval(pomoTimerInt);
            pomoTimerInt = null;
            btn.textContent = 'INICIAR';
            onPomoCycleFinished();
          }
        }, 1000);
      }
    }

    function skipPomoCycle() {
      if (confirm('Deseja pular este ciclo?')) {
        if (pomoTimerInt) { clearInterval(pomoTimerInt); pomoTimerInt = null; }
        document.getElementById('btn-pomo-main').textContent = 'INICIAR';
        advancePomoCycle(false);
      }
    }

    function finishCurrentBlockNow() {
      // Permite contabilizar o tempo transcorrido no bloco atual
      const transcorridoSeg = pomoInitialSeconds - pomoTotalSeconds;
      const transcorridoMin = Math.round(transcorridoSeg / 60);
      
      if (transcorridoMin < 1) {
        alert('Você precisa realizar ao menos 1 minuto de foco para registrar!');
        return;
      }

      if (pomoTimerInt) { clearInterval(pomoTimerInt); pomoTimerInt = null; }
      document.getElementById('btn-pomo-main').textContent = 'INICIAR';

      if (currentPomoMode === 'focus') {
        const taskName = document.getElementById('pomo-task-name').value.trim() || 'Bloco Pomodoro';
        registrarSessaoEstudoNoHistorico(transcorridoMin, taskName);
        showToast(`🎉 <strong>${transcorridoMin} min</strong> de foco contabilizados no seu relatório!`);
      }
      advancePomoCycle(true);
    }

    function onPomoCycleFinished() {
      if (pomoSettings.sound) playZenChime();

      if (currentPomoMode === 'focus') {
        const durationMin = Math.round(pomoInitialSeconds / 60);
        const taskName = document.getElementById('pomo-task-name').value.trim() || 'Bloco Pomodoro';
        registrarSessaoEstudoNoHistorico(durationMin, taskName);
        showToast(`🎉 <strong>Bloco de ${durationMin} min</strong> concluído e contabilizado no seu relatório!`);
        advancePomoCycle(true);
      } else {
        showToast('⏰ Pausa concluída! Hora de focar.');
        setPomoMode('focus');
      }
    }

    function advancePomoCycle(completed) {
      if (currentPomoMode === 'focus') {
        if (pomoRound >= 4) {
          pomoRound = 1;
          setPomoMode('long');
        } else {
          pomoRound++;
          setPomoMode('short');
        }
      } else {
        setPomoMode('focus');
      }
    }

    // Modal Configurações
    function openPomoSettingsModal() {
      document.getElementById('cfg-pomo-focus').value = pomoSettings.focusMin;
      document.getElementById('cfg-pomo-short').value = pomoSettings.shortMin;
      document.getElementById('cfg-pomo-long').value = pomoSettings.longMin;
      document.getElementById('cfg-pomo-sound').checked = pomoSettings.sound;
      document.getElementById('modal-pomo-settings').classList.add('open');
    }

    function closePomoSettingsModal() {
      document.getElementById('modal-pomo-settings').classList.remove('open');
    }

    function salvarConfigPomodoro() {
      const f = parseInt(document.getElementById('cfg-pomo-focus').value) || 25;
      const s = parseInt(document.getElementById('cfg-pomo-short').value) || 5;
      const l = parseInt(document.getElementById('cfg-pomo-long').value) || 15;
      const snd = document.getElementById('cfg-pomo-sound').checked;

      pomoSettings = { focusMin: f, shortMin: s, longMin: l, sound: snd };
      localStorage.setItem('pomodoroConfig', JSON.stringify(pomoSettings));
      closePomoSettingsModal();
      initPomoDisplay();
      showToast('⚙️ Configurações do Pomodoro salvas com sucesso!');
    }

    // Som de sino relaxante ao terminar (Web Audio)
    function playZenChime() {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.3); // A5
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.8);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 1.8);
      } catch(e){}
    }

    // Gerador de Som de Chuva Minimalista
    let rainAudioCtx = null;
    let rainNode = null;
    let isRainOn = false;

    function toggleRainSound() {
      const icon = document.getElementById('rain-icon');
      const txt = document.getElementById('rain-txt');
      const btn = document.getElementById('btn-sound-toggle');

      if (!isRainOn) {
        rainAudioCtx = rainAudioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const buf = rainAudioCtx.createBuffer(1, rainAudioCtx.sampleRate * 2, rainAudioCtx.sampleRate);
        const out = buf.getChannelData(0);
        let last = 0;
        for (let i = 0; i < out.length; i++) {
          const w = Math.random() * 2 - 1;
          out[i] = (last + (0.02 * w)) / 1.02;
          last = out[i];
          out[i] *= 3.5;
        }
        rainNode = rainAudioCtx.createBufferSource();
        rainNode.buffer = buf;
        rainNode.loop = true;
        const gain = rainAudioCtx.createGain();
        gain.gain.value = 0.08;
        rainNode.connect(gain);
        gain.connect(rainAudioCtx.destination);
        rainNode.start();
        isRainOn = true;
        btn.classList.add('active');
        icon.textContent = '🌧️';
        txt.textContent = 'Chuva Zen: Ligada';
      } else {
        if (rainNode) { rainNode.stop(); rainNode.disconnect(); rainNode = null; }
        isRainOn = false;
        btn.classList.remove('active');
        icon.textContent = '🌧️';
        txt.textContent = 'Chuva Zen: Desligada';
      }
    }

    /* ========================================================
       5. SISTEMA DE PERSISTÊNCIA & ATIVIDADES (localStorage)
       ======================================================== */
    function getActivities() {
      try { return JSON.parse(localStorage.getItem('focusActivities') || '[]'); }
      catch(e) { return []; }
    }
    function saveActivities(arr) { localStorage.setItem('focusActivities', JSON.stringify(arr)); }

    function getTodayString() {
      return new Date().toISOString().split('T')[0];
    }

    // Inicializar campos de data
    (function initDatePickers(){
      const today = getTodayString();
      const td = document.getElementById('train-date');
      const sd = document.getElementById('study-date');
      if (td) td.value = today;
      if (sd) sd.value = today;
    })();

    // Contabiliza sessão de estudo gerada pelo Pomodoro
    function registrarSessaoEstudoNoHistorico(minutos, nota) {
      const acts = getActivities();
      acts.push({
        type: 'study',
        date: getTodayString(),
        duration: minutos,
        note: nota,
        timestamp: Date.now()
      });
      saveActivities(acts);
    }

    function salvarEstudoManual() {
      const date = document.getElementById('study-date').value || getTodayString();
      const dur = parseInt(document.getElementById('study-duration').value);
      const note = document.getElementById('study-note').value.trim() || 'Estudo manual';
      if (!dur || dur < 1) { alert('Informe a duração em minutos.'); return; }

      const acts = getActivities();
      acts.push({ type: 'study', date, duration: dur, note, timestamp: Date.now() });
      saveActivities(acts);
      renderActivityHistories();
      showToast('✅ Sessão de estudo registrada!');
      document.getElementById('study-note').value = '';
    }

    function salvarTreinoManual() {
      const date = document.getElementById('train-date').value || getTodayString();
      const dur = parseInt(document.getElementById('train-duration').value);
      const type = document.getElementById('train-type').value;
      if (!dur || dur < 1) { alert('Informe a duração em minutos.'); return; }

      const acts = getActivities();
      acts.push({ type: 'train', date, duration: dur, note: type, timestamp: Date.now() });
      saveActivities(acts);
      renderActivityHistories();
      showToast('🏋️ Treino registrado com sucesso!');
    }

    function deletarAtividade(ts) {
      let acts = getActivities().filter(a => a.timestamp !== ts);
      saveActivities(acts);
      renderActivityHistories();
    }

    function limparTodoHistorico() {
      if (confirm('Tem certeza que deseja apagar todo o histórico de atividades e relatórios?')) {
        localStorage.removeItem('focusActivities');
        renderActivityHistories();
        renderReportAnalytics();
        showToast('🗑️ Histórico limpo.');
      }
    }

    function formatDateFriendly(ds) {
      if (!ds) return '';
      const [y, m, d] = ds.split('-');
      return `${d}/${m}`;
    }

    function renderActivityHistories() {
      const acts = getActivities();
      const sList = document.getElementById('study-history-list');
      const tList = document.getElementById('train-history-list');

      const studies = acts.filter(a => a.type === 'study').sort((a,b) => b.timestamp - a.timestamp).slice(0, 10);
      const trains = acts.filter(a => a.type === 'train').sort((a,b) => b.timestamp - a.timestamp).slice(0, 10);

      if (sList) {
        sList.innerHTML = studies.length === 0
          ? '<div style="color:var(--text-tertiary);font-size:0.75rem;text-align:center;padding:12px;">Nenhum estudo ainda. Execute um bloco no <strong>Pomodoro</strong>! ⏱️</div>'
          : studies.map(s => `
            <div class="history-row">
              <span class="history-date">${formatDateFriendly(s.date)}</span>
              <span class="history-dur">${s.duration} min</span>
              <span class="history-desc">${s.note}</span>
              <button onclick="deletarAtividade(${s.timestamp})" style="background:none;border:none;color:var(--text-tertiary);cursor:pointer;" title="Excluir">✕</button>
            </div>
          `).join('');
      }

      if (tList) {
        tList.innerHTML = trains.length === 0
          ? '<div style="color:var(--text-tertiary);font-size:0.75rem;text-align:center;padding:12px;">Nenhum treino registrado. Registre seu exercício de hoje! 🏋️</div>'
          : trains.map(t => `
            <div class="history-row">
              <span class="history-date">${formatDateFriendly(t.date)}</span>
              <span class="history-dur" style="color:var(--success);">${t.duration} min</span>
              <span class="history-desc">${t.note}</span>
              <button onclick="deletarAtividade(${t.timestamp})" style="background:none;border:none;color:var(--text-tertiary);cursor:pointer;" title="Excluir">✕</button>
            </div>
          `).join('');
      }
    }

    /* ========================================================
       6. RELATÓRIO ANALYTICS & STREAKS
       ======================================================== */
    function renderReportAnalytics() {
      const acts = getActivities();
      const studies = acts.filter(a => a.type === 'study');
      const trains = acts.filter(a => a.type === 'train');

      const studyMin = studies.reduce((s, a) => s + a.duration, 0);
      const trainMin = trains.reduce((s, a) => s + a.duration, 0);
      const uniqueTrainDays = new Set(trains.map(a => a.date)).size;

      document.getElementById('rpt-study-hours').textContent = (studyMin / 60).toFixed(1) + 'h';
      document.getElementById('rpt-train-days').textContent = uniqueTrainDays;
      document.getElementById('rpt-study-sessions').textContent = studies.length;
      document.getElementById('rpt-train-hours').textContent = (trainMin / 60).toFixed(1) + 'h';

      // Streaks
      const sStreak = calcStreak(studies);
      const tStreak = calcStreak(trains);

      document.getElementById('rpt-study-streak').textContent = sStreak;
      document.getElementById('rpt-train-streak').textContent = tStreak;
      document.getElementById('rpt-study-fire').textContent = sStreak >= 3 ? '🔥' : '';
      document.getElementById('rpt-train-fire').textContent = tStreak >= 3 ? '🔥' : '';

      renderTiles('study-tiles', studies);
      renderTiles('train-tiles', trains);
      renderWeeklyBars(acts);
    }

    function calcStreak(entries) {
      const dates = [...new Set(entries.map(e => e.date))].sort().reverse();
      if (dates.length === 0) return 0;

      const today = getTodayString();
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

      if (dates[0] !== today && dates[0] !== yesterday) return 0;

      let count = 1;
      for (let i = 1; i < dates.length; i++) {
        const d1 = new Date(dates[i-1] + 'T12:00:00');
        const d2 = new Date(dates[i] + 'T12:00:00');
        const diffDays = Math.round((d1 - d2) / 86400000);
        if (diffDays === 1) count++;
        else break;
      }
      return count;
    }

    function renderTiles(targetId, entries) {
      const el = document.getElementById(targetId);
      const dateSet = new Set(entries.map(e => e.date));
      const today = new Date();
      let html = '';
      for (let i = 13; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const ds = d.toISOString().split('T')[0];
        const isAct = dateSet.has(ds);
        const isTod = i === 0;
        html += `<div class="streak-tile${isAct ? ' active' : ''}${isTod ? ' today' : ''}" title="${formatDateFriendly(ds)}: ${isAct ? 'Ativo' : 'Sem registro'}"></div>`;
      }
      el.innerHTML = html;
    }

    function renderWeeklyBars(acts) {
      const area = document.getElementById('weekly-bars-area');
      const today = new Date();
      const days = ['Dom','Seg','Ter','Qua','Qui','Sex','Sáb'];
      let maxMin = 1;
      const data = [];

      for (let i = 6; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const ds = d.toISOString().split('T')[0];
        const studySum = acts.filter(a => a.type === 'study' && a.date === ds).reduce((s, a) => s + a.duration, 0);
        const trainSum = acts.filter(a => a.type === 'train' && a.date === ds).reduce((s, a) => s + a.duration, 0);
        const total = studySum + trainSum;
        if (total > maxMin) maxMin = total;
        data.push({ label: days[d.getDay()], study: studySum, train: trainSum, isToday: i === 0 });
      }

      area.innerHTML = data.map(d => {
        const sH = Math.round((d.study / maxMin) * 90);
        const tH = Math.round((d.train / maxMin) * 90);
        const total = d.study + d.train;
        return `
          <div class="chart-col">
            <span class="col-val">${total > 0 ? total + 'm' : ''}</span>
            <div class="col-bars-stack">
              <div class="col-seg study" style="height:${sH}px;"></div>
              <div class="col-seg train" style="height:${tH}px;"></div>
            </div>
            <span class="col-label" style="${d.isToday ? 'color:var(--accent);font-weight:700;' : ''}">${d.label}</span>
          </div>
        `;
      }).join('');
    }

    /* ========================================================
       7. COACH ZEN (CHATBOT GEMINI INTEGRATION)
       ======================================================== */
    function toggleKeyBar() {
      const bar = document.getElementById('key-bar');
      bar.style.display = bar.style.display === 'none' ? 'block' : 'none';
    }

    function salvarGeminiKeyManual() {
      const val = document.getElementById('gemini-key-input').value.trim();
      geminiApiKey = val;
      localStorage.setItem('GEMINI_API_KEY', val);
      toggleKeyBar();
      showToast(val ? '🟢 Chave Gemini salva!' : 'Modo padrão restaurado.');
    }

    function appendChatMessage(role, text) {
      const box = document.getElementById('chat-messages');
      const row = document.createElement('div');
      row.className = `chat-bubble-row ${role}`;
      row.innerHTML = `<div class="chat-bubble ${role}">${text}</div>`;
      box.appendChild(row);
      box.scrollTop = box.scrollHeight;
    }

    function sendQuickPrompt(type) {
      const prompts = {
        'sos': '🚨 SOS: Vontade incontrolável de rolar o feed agora!',
        'ansiedade': '⚡ Como conter a abstinência e inquietação ao ficar longe do celular?',
        'sono': '😴 Dormi mal, como conseguir estudar hoje com foco?'
      };
      if (prompts[type]) executeChat(prompts[type]);
    }

    function handleChatSend() {
      const inp = document.getElementById('chat-input-text');
      const txt = inp.value.trim();
      if (!txt) return;
      inp.value = '';
      executeChat(txt);
    }

    async function executeChat(msg) {
      appendChatMessage('user', msg);

      const context = {
        horas_redes_sociais: parseFloat(document.getElementById('input-horas').value),
        busca_dopamina_scroll: parseInt(document.getElementById('input-dopamina').value),
        ansiedade_sem_celular: parseInt(document.getElementById('input-ansiedade').value),
        problemas_sono: parseInt(document.getElementById('input-sono').value),
        dificuldade_concentracao: parseInt(document.getElementById('input-concentracao').value),
        probabilidade: currentProb,
        status: currentProb < 35 ? 'FOCO_PRESERVADO' : (currentProb < 60 ? 'RISCO_MODERADO' : 'ALTO_RISCO')
      };

      appendChatMessage('bot', '<span style="opacity:0.6;">Digitando resposta... 🧠</span>');
      const box = document.getElementById('chat-messages');
      const loading = box.lastElementChild;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg, context: context, api_key: geminiApiKey })
        });
        const data = await res.json();
        loading.remove();
        appendChatMessage('bot', data.reply || 'Sem resposta do coach.');
      } catch(err) {
        loading.remove();
        appendChatMessage('bot', '<strong>Dica Rápida:</strong> Para romper o atrito, vá à aba <strong>Pomodoro</strong> e faça apenas 5 minutos de foco ininterrupto.');
      }
    }

    // Inicialização geral
    recalcular();
    initPomoDisplay();
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
