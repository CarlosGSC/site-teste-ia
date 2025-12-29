import streamlit as st
import pandas as pd
import time
from io import BytesIO
import re

# --- IMPORTAÇÕES ---
from google import genai
from google.genai import types

from openai import OpenAI
import anthropic

# --- Configuração da Página ---
st.set_page_config(page_title="LLM Benchmark: Native Thinking", layout="wide")

st.title("🧠 LLM Benchmark: Native Thinking Control")
st.markdown("""
**Controle Total:** Escolha entre Thinking Dinâmico, Manual (Budget Fixo) ou Desativado.
**Requisito:** Requer a biblioteca `pip install google-genai`.
""")

# --- Configurações de Modelos ---
MODELS_CONFIG = {
    "gpt-5.2": {
        "provider": "OpenAI",
        "input": 1.75, "output": 14.00, 
        "type": "reasoning_native" 
    },
    "gpt-5.1": {
        "provider": "OpenAI",
        "input": 1.25, "output": 10.00, 
        "type": "reasoning_native" 
    },
    "gpt-5-mini": {
        "provider": "OpenAI",
        "input": 0.25, "output": 2.00, 
        "type": "reasoning_native" 
    },
    "gpt-5-nano": {
        "provider": "OpenAI",
        "input": 0.05, "output": 0.40, 
        "type": "reasoning_native" 
    },
    "gemini-3-pro-preview": {
        "provider": "Google", 
        "input": 2, "output": 12, 
        "type": "standard" 
    },
    "gemini-3-flash-preview": {
        "provider": "Google", 
        "input": 0.5, "output": 3, 
        "type": "native_thinking_capable" 
    },
    "gemini-2.5-pro": {
        "provider": "Google", 
        "input": 1.25, "output": 10, 
        "type": "standard" 
    },
    "gemini-2.5-flash-lite": {
        "provider": "Google", 
        "input": 0.1, "output": 0.4, 
        "type": "native_thinking_capable" 
    },
    "gemini-2.5-flash": {
        "provider": "Google", 
        "input": 0.3, "output": 2.5,
        "type": "native_thinking_capable"
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "Anthropic", 
        "input": 3.00, "output": 15.00,
        "type": "standard"
    },
    "grok-beta": {
        "provider": "xAI", 
        "input": 5.00, "output": 15.00,
        "type": "standard"
    }
}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações de Raciocínio")
    
    st.markdown("### Controle do Gemini 2.5")
    thinking_mode = st.radio(
        "Modo de Thinking:",
        options=["Desativado", "Dinâmico (Automático)", "Manual (Orçamento Fixo)"],
        index=1,
        help="Dinâmico ajusta conforme a dificuldade (-1). Manual fixa um limite."
    )
    
    budget_value = 0
    if thinking_mode == "Manual (Orçamento Fixo)":
        budget_value = st.slider("Tokens de Pensamento:", 128, 8192, 1024, 128)
    elif thinking_mode == "Dinâmico (Automático)":
        budget_value = -1 
    else:
        budget_value = 0 
    
    st.caption(f"Budget enviado: `{budget_value}`")

    st.divider()
    st.header("🔑 Credenciais")
    openai_key = st.text_input("OpenAI API Key", type="password")
    google_key = st.text_input("Google AI Key", type="password")
    anthropic_key = st.text_input("Anthropic Key", type="password")
    xai_key = st.text_input("xAI Key", type="password")

    st.divider()
    
    selected_models = st.multiselect(
        "Escolha os competidores:",
        options=list(MODELS_CONFIG.keys()),
        default=["gemini-2.5-flash-lite"]
    )

# --- FUNÇÕES DE API ---

def call_openai_style(model, prompt, api_key, base_url=None):
    client = OpenAI(api_key=api_key, base_url=base_url)
    config = MODELS_CONFIG.get(model, {})
    is_o1 = config.get("type") == "reasoning_native"
    
    try:
        if is_o1:
            messages = [{"role": "user", "content": f"Responda apenas a letra da alternativa correta. Questão: {prompt}"}]
            kwargs = {} 
        else:
            messages = [
                {"role": "system", "content": "Responda apenas com a alternativa correta (Ex: A, B, C...)."},
                {"role": "user", "content": prompt}
            ]
            kwargs = {"temperature": 0}

        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content
        usage = response.usage
        
        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
             reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
        
        # Garante que não é None
        reasoning_tokens = reasoning_tokens or 0
        
        return content, usage.prompt_tokens, usage.completion_tokens, reasoning_tokens, ""
        
    except Exception as e:
        return f"Erro: {str(e)}", 0, 0, 0, ""

def call_google(model_name, prompt, api_key, budget_param):
    try:
        client = genai.Client(api_key=api_key)
        config_info = MODELS_CONFIG.get(model_name, {})
        
        thinking_config = None
        
        if config_info.get("type") == "native_thinking_capable":
            if budget_param == 0:
                thinking_config = types.ThinkingConfig(thinking_budget=0)
            else:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True, 
                    thinking_budget=budget_param
                )
        
        final_config = types.GenerateContentConfig(
            temperature=0 if budget_param == 0 else None,
            thinking_config=thinking_config
        )

        response = client.models.generate_content(
            model=model_name,
            contents=f"Responda apenas com a letra da alternativa correta. Questão: {prompt}",
            config=final_config
        )
        
        final_text = ""
        thought_summary = ""
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.thought: 
                    thought_summary += part.text + "\n"
                else:
                    final_text += part.text
        
        usage = response.usage_metadata
        in_tok = usage.prompt_token_count
        out_tok = usage.candidates_token_count
        
        # --- CORREÇÃO DO ERRO AQUI ---
        # Adicionei "or 0" para garantir que se vier None, vira 0 (Inteiro)
        think_tok = getattr(usage, 'thoughts_token_count', 0) or 0
        
        return final_text, in_tok, out_tok, think_tok, thought_summary

    except Exception as e:
        return f"Erro Google API: {str(e)}", 0, 0, 0, ""

def call_anthropic(model_name, prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model_name, max_tokens=100, temperature=0,
            system="Responda apenas com a alternativa correta.",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text, message.usage.input_tokens, message.usage.output_tokens, 0, ""
    except Exception as e:
        return f"Erro: {str(e)}", 0, 0, 0, ""

# --- Roteador ---
def get_model_response(model_name, prompt, budget_setting):
    provider = MODELS_CONFIG[model_name]["provider"]
    
    if provider == "OpenAI":
        if not openai_key: return "Erro: Falta Key", 0, 0, 0, ""
        return call_openai_style(model_name, prompt, openai_key)
        
    elif provider == "Google":
        if not google_key: return "Erro: Falta Key", 0, 0, 0, ""
        return call_google(model_name, prompt, google_key, budget_param=budget_setting)
        
    elif provider == "Anthropic":
        if not anthropic_key: return "Erro: Falta Key", 0, 0, 0, ""
        return call_anthropic(model_name, prompt, anthropic_key)
    
    elif provider == "xAI":
        if not xai_key: return "Erro: Falta Key", 0, 0, 0, ""
        return call_openai_style(model_name, prompt, xai_key, base_url="https://api.x.ai/v1")
    
    return "Modelo desconhecido", 0, 0, 0, ""

# --- Interface Principal ---
uploaded_file = st.file_uploader("📂 Upload CSV (Colunas: 'question', 'answer')", type=["csv"])

if uploaded_file and selected_models:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower() for c in df.columns]

    if 'question' in df.columns and 'answer' in df.columns:
        if st.button("🚀 Iniciar Teste"):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_steps = len(df) * len(selected_models)
            current_step = 0

            for idx, row in df.iterrows():
                question = row['question']
                answer = str(row['answer']).strip().upper()

                for model in selected_models:
                    current_step += 1
                    status_text.markdown(f"**Testando:** {model} | Questão {idx+1}")
                    
                    start_time = time.time()
                    
                    resp_text, in_tok, out_tok, think_tok, think_text = get_model_response(model, question, budget_value)
                    
                    # Garantia extra para evitar o erro, caso o retorno da função ainda falhe
                    if think_tok is None: think_tok = 0
                    if in_tok is None: in_tok = 0
                    if out_tok is None: out_tok = 0
                    
                    end_time = time.time()
                    
                    clean_text = resp_text.strip().upper().replace(".", "")[:1]
                    acertou = "✅ Sim" if clean_text == answer else "❌ Não"
                    if "Erro" in resp_text: acertou = "⚠️ Erro API"

                    conf = MODELS_CONFIG[model]
                    
                    # Cálculo de Custo com sua fórmula
                    if think_tok > 0:
                        cost = (in_tok / 1_000_000 * conf['input']) + ((out_tok + think_tok) / 1_000_000 * conf['output'])
                    else:
                        cost = (in_tok / 1_000_000 * conf['input']) + (out_tok / 1_000_000 * conf['output'])

                    results.append({
                        "Modelo": model,
                        "Questão": idx + 1,
                        "Gabarito": answer,
                        "Resposta": clean_text,
                        "Assertividade": acertou,
                        "Tempo (s)": round(end_time - start_time, 2),
                        "Custo ($)": cost,
                        "Tokens Entrada": in_tok,
                        "Tokens Saída": out_tok,
                        "Thinking Tokens": think_tok,
                        "Resumo Pensamento": think_text[:100] + "..." if think_text else "" 
                    })
                    
                    progress_bar.progress(current_step / total_steps)

            status_text.success("Testes Finalizados!")
            
            # --- Relatório ---
            res_df = pd.DataFrame(results)
            
            summary = res_df.groupby("Modelo").agg(
                Acertos=('Assertividade', lambda x: (x == '✅ Sim').sum()),
                Total=('Questão', 'count'),
                Tempo_Medio=('Tempo (s)', 'mean'),
                Custo_Total=('Custo ($)', 'sum'),
                Thinking_Medio=('Thinking Tokens', 'mean'),
                Entrada_Media=('Tokens Entrada', 'mean'),
                Saida_Media=('Tokens Saída', 'mean')
            ).reset_index()
            
            summary['Taxa Acerto (%)'] = (summary['Acertos'] / summary['Total']) * 100

            st.subheader("📊 Resultados Consolidados")
            st.dataframe(summary.style.format({
                "Custo_Total": "${:.6f}", 
                "Taxa Acerto (%)": "{:.1f}%",
                "Thinking_Medio": "{:.0f}",
                "Entrada_Media": "{:.0f}",
                "Saida_Media": "{:.0f}"
            }))

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, sheet_name='Detalhado', index=False)
                summary.to_excel(writer, sheet_name='Resumo', index=False)
                wb = writer.book
                fmt = wb.add_format({'num_format': '$0.000000'})
                writer.sheets['Resumo'].set_column('E:E', 15, fmt)
            
            st.download_button("📥 Baixar Excel", buffer.getvalue(), "benchmark_thinking.xlsx")

    else:
        st.error("CSV inválido. Colunas necessárias: 'question', 'answer'.")