# ==============================================================
# Predição de Abandono de Curso (Streamlit + CatBoost)
# >>> 9 features + Q03 condicional | Classe 0=Abandono, Classe 1=Conclusão
# >>> Com: banners, JSON de entrada, barras com limiares, resumo clínico e SHAP
# ==============================================================

from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd

# --------------------------------------------------------------
# Config da página
# --------------------------------------------------------------
st.set_page_config(page_title="🎓 Predição de Abandono de Curso",
                   page_icon="🎓", layout="wide")

# --------------------------------------------------------------
# Caminhos e utilitários
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# 🔁 Ajuste para o caminho do modelo enxuto salvo (model.cbm)
MODEL_FILE = BASE_DIR / "model.cbm"

ASSETS = BASE_DIR / "assets"
LOGO = (ASSETS / "logo.png") if (ASSETS / "logo.png").exists() else None

def metric_card(title: str, value: str, help: str | None = None) -> None:
    st.metric(title, value, help=help)

def nice_label(name: str) -> str:
    """rótulos de exibição amigáveis para o SHAP/JSON"""
    MAP = {
        "status_formulario_inicial": "Status do formulário inicial",
        "matricula_ingresso_inicio_dias": "Dias desde o ingresso",
        "aluno_profissional_profissao_descricao": "Profissão",
        "hist_12m_conclusoes": "Conclusões (12m)",
        "hist_total_conclusoes": "Conclusões (total)",
        "hist_total_abandonos": "Abandonos (total)",
        "form_inicial_atual_q03_nivel_conhecimento": "Q03: nível de conhecimento",
        "aluno_acesso_acesso_uf": "UF de acesso",
        "aluno_profissional_escolaridade_descricao": "Escolaridade",
    }
    return MAP.get(name, name)

# --------------------------------------------------------------
# ORDEM EXATA do treino  ✅ (não alterar)
# --------------------------------------------------------------
FEATURES = [
    "status_formulario_inicial",                 # cat
    "matricula_ingresso_inicio_dias",            # num
    "aluno_profissional_profissao_descricao",    # cat
    "hist_12m_conclusoes",                       # num
    "hist_total_conclusoes",                     # num
    "hist_total_abandonos",                      # num
    "form_inicial_atual_q03_nivel_conhecimento", # num (0–10, condicional)
    "aluno_acesso_acesso_uf",                    # cat
    "aluno_profissional_escolaridade_descricao", # cat
]
CATEGORICAL_FEATURES = [
    "status_formulario_inicial",
    "aluno_profissional_profissao_descricao",
    "aluno_acesso_acesso_uf",
    "aluno_profissional_escolaridade_descricao",
]
CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]

# --------------------------------------------------------------
# Carregar modelo
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: Path) -> CatBoostClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    m = CatBoostClassifier()
    m.load_model(str(path))
    return m

try:
    model = load_model(MODEL_FILE)
    st.success("✅ Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"Falha ao carregar modelo CatBoost: {e}")
    st.stop()

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_column_width=True)
    st.header("Menu")
    st.page_link("app.py", label="Predição de Abandono de Curso", icon="📈")
    st.page_link("pages/explain.py", label="Explicação do Modelo", icon="📙")

# --------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(90deg,#0f4c81,#002b55);
                padding:28px;border-radius:14px;margin-bottom:22px'>
      <h1 style='color:#fff;margin:0'>🎓 Predição de Abandono de Curso</h1>
      <p style='color:#dfe9f3;margin:6px 0 0 0'>
         Aplicação de apoio à decisão. O resultado é probabilístico.
      </p>
    </div>
    """, unsafe_allow_html=True
)

# ==============================================================
# FORMULÁRIO – seções na ordem que você pediu
# ==============================================================

# ---------------------- DADOS PROFISSIONAIS -------------------
st.subheader("Dados Profissionais")

escolaridade_opts = [
    "Ensino Fundamental", "Ensino Médio", "Técnico de Nível Médio", "Graduação",
    "Graduação Tecnológica", "Especialização", "Mestrado Profissional",
    "Mestrado Acadêmico", "Residência Médica", "Residência Multiprofissional",
    "Doutorado", "Não identificado",
]
uf_opts = [
    "SP", "MG", "PE", "PR", "RJ", "BA", "RS", "CE", "SC", "GO", "ES", "DF", "PA", "MS",
    "PB", "AM", "MA", "SE", "MT", "RO", "RN", "AL", "PI", "TO", "AC", "AP", "RR",
    "Não identificado",
]
profissao_opts = [
    "Estudante", "Enfermeiro", "Médico", "Técnico de Enfermagem", "Farmacêutico",
    "Auxiliar de Enfermagem", "Biomédico", "Dentista", "Fisioterapeuta", "Biólogo",
    "Assistente Social", "Nutricionista", "Psicólogo", "Médico Veterinário",
    "Profissionais de Educação Física", "Fonoaudiólogo", "Terapeuta Ocupacional",
    "Agente Comunitário de Saúde", "Outros", "Não identificado",
]

cE, cU, cP = st.columns(3)
with cE:
    aluno_profissional_escolaridade_descricao = st.selectbox(
        "Escolaridade", escolaridade_opts, index=3
    )
with cU:
    aluno_acesso_acesso_uf = st.selectbox("UF de acesso", uf_opts, index=0)
with cP:
    aluno_profissional_profissao_descricao = st.selectbox(
        "Profissão", profissao_opts, index=0
    )

st.markdown("---")

# --------------- DADOS HISTÓRICO EDUCACIONAIS -----------------
st.subheader("Dados Histórico Educacionais")
c1, c2, c3 = st.columns(3)
with c1:
    hist_total_conclusoes = st.slider("Conclusões (total)", 0, 200, 0, 1)
with c2:
    hist_12m_conclusoes = st.slider("Conclusões (12 meses)", 0, 50, 0, 1)
with c3:
    hist_total_abandonos = st.slider("Abandonos (total)", 0, 200, 0, 1)

st.markdown("---")

# --------------------- DADOS DO CURSO ATUAL -------------------
st.subheader("Dados do Curso Atual")

cD, cS = st.columns(2)
with cD:
    matricula_ingresso_inicio_dias = st.slider(
        "Dias desde o ingresso (matrícula)", 0, 365, 30, 1
    )
with cS:
    status_formulario_inicial = st.selectbox(
        "Status do formulário inicial",
        ["Respondeu formulário", "Não respondeu"],
        index=0,
    )

# Q03 condicional
if status_formulario_inicial == "Respondeu formulário":
    form_inicial_atual_q03_nivel_conhecimento = st.slider(
        "Q03: nível de conhecimento (0–10)", 0.0, 10.0, 5.0, 0.5
    )
else:
    st.info("Q03 não disponível (formulário não respondido). Valor assumido: 0.")
    form_inicial_atual_q03_nivel_conhecimento = 0.0

st.markdown("---")

# --------------------- Limiar e botões ------------------------
# Limiar refere-se à PROBABILIDADE DE ABANDONO (classe 0)
th_map = {"0.30 (triagem ampla)": 0.30, "0.50 (padrão)": 0.50, "0.70 (mais precisão)": 0.70}
choice = st.radio("Limiar para ALERTA de abandono (classe 0)",
                  list(th_map.keys()), index=1, horizontal=True)
LIMIAR = th_map[choice]

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    submit = st.button("Enviar", type="primary", use_container_width=True)
with col_btn2:
    reset = st.button("Nova simulação 🔁", use_container_width=True)
    if reset:
        st.experimental_rerun()

# --------------------------------------------------------------
# Inferência
# --------------------------------------------------------------
if submit:
    # montagem da linha PARA O MODELO (ordem EXATA de FEATURES)
    row = [
        status_formulario_inicial,                          # 1
        int(matricula_ingresso_inicio_dias),                # 2
        aluno_profissional_profissao_descricao,             # 3
        int(hist_12m_conclusoes),                           # 4
        int(hist_total_conclusoes),                         # 5
        int(hist_total_abandonos),                          # 6
        float(form_inicial_atual_q03_nivel_conhecimento),   # 7
        aluno_acesso_acesso_uf,                             # 8
        aluno_profissional_escolaridade_descricao,          # 9
    ]
    assert len(row) == len(FEATURES), "Número de entradas difere do esperado."

    # banner + JSON dos dados enviados (com rótulos amigáveis)
    st.success("Dados enviados com sucesso!")
    st.json({nice_label(k): v for k, v in zip(FEATURES, row)})

    X_pool = Pool(data=[row], cat_features=CAT_IDX)
    proba = model.predict_proba(X_pool)[0]   # [P(classe=0), P(classe=1)]
    prob_abandono  = float(proba[0])  # classe 0
    prob_concluir  = float(proba[1])  # classe 1

    # ----------------------------------------------------------
    # Resultado da predição — barras com limiares
    # ----------------------------------------------------------
    st.markdown("## Resultado da Predição:")
    st.markdown("**Distribuição das Probabilidades das Classes**")

    # cores: verde para a classe mais provável, vermelho para a outra
    if prob_abandono >= prob_concluir:
        colors = ["#2ecc71", "#e74c3c"]  # 0 verde, 1 vermelho
    else:
        colors = ["#e74c3c", "#2ecc71"]  # 0 vermelho, 1 verde

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Classe 0 - Abandono", "Classe 1 - Conclusão"],
        y=[prob_abandono, prob_concluir],
        marker_color=colors,
        text=[f"{prob_abandono:.2%}", f"{prob_concluir:.2%}"],
        textposition="inside"
    ))
    # linhas de limiar: abandono = LIMIAR; conclusão = 1 - LIMIAR (apenas referencial)
    fig.add_hline(y=LIMIAR, line_dash="dash", line_color="#1f77b4",
                  annotation_text=f"Limiar Classe 0 ({LIMIAR:.0%})",
                  annotation_position="top left")
    fig.add_hline(y=1.0 - LIMIAR, line_dash="dash", line_color="#9467bd",
                  annotation_text=f"Limiar Classe 1 ({(1-LIMIAR):.0%})",
                  annotation_position="top right")
    fig.update_layout(
        yaxis=dict(range=[0, 1], title="Probabilidade"),
        xaxis=dict(title="Classes"),
        bargap=0.25, plot_bgcolor="rgba(0,0,0,0)", height=420,
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    # card de status
    alerta_abandono = (prob_abandono >= LIMIAR)
    if alerta_abandono:
        st.error(f"🔔 **Alerta:** risco de abandono (classe 0) = {prob_abandono:.1%} ≥ limiar {LIMIAR:.0%}")
    else:
        st.success(f"✅ Baixo risco de abandono (classe 0) = {prob_abandono:.1%} < limiar {LIMIAR:.0%}")

    # ----------------------------------------------------------
    # Resumo clínico para decisão (estilo seu outro app)
    # ----------------------------------------------------------
    st.markdown("## Resumo clínico para decisão")
    if alerta_abandono:
        st.markdown(f"**Classe predita:** <span style='color:#c0392b'>Abandono (Classe 0)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Classe predita:** <span style='color:#27ae60'>Conclusão (Classe 1)</span>", unsafe_allow_html=True)
    st.markdown(f"- **Probabilidade estimada:** {prob_concluir:.1%} para **concluir** | {prob_abandono:.1%} para **abandonar** "
                f"(limiar de alerta: {LIMIAR:.0%}).")

    # ----------------------------------------------------------
    # Explicação SHAP (para a observação atual)
    # ----------------------------------------------------------
    # CatBoost retorna SHAP em "raw score" (log-odds), mas serve para comparar impactos.
    shap_vals = model.get_feature_importance(data=X_pool, type="ShapValues")
    # shape: (1, n_features+1) -> último é o expected_value (base)
    phi = shap_vals[0, :-1]
    base = shap_vals[0, -1]

    # ordenar por |impacto|
    order = np.argsort(np.abs(phi))[::-1]
    top_n = min(12, len(FEATURES))
    order = order[:top_n]

    feats_sorted = [FEATURES[i] for i in order]
    phi_sorted = phi[order]

    # bullets com setas ↑/↓ risco (negativo -> p/ classe 0; positivo -> p/ classe 1)
    st.markdown("**Principais fatores que influenciaram esta predição:**")
    bullets = []
    for f, v in zip(feats_sorted[:5], phi_sorted[:5]):
        seta = "↑ risco" if v < 0 else "↓ risco"
        bullets.append(f"- **{nice_label(f)}** — impacto (SHAP): {v:+.3f} → {seta}")
    st.markdown("\n".join(bullets))

    # gráfico horizontal dos impactos
    bar_colors = ["#c0392b" if v < 0 else "#1abc9c" for v in phi_sorted]  # vermelhos (abandono), verdes (conclusão)
    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=phi_sorted[::-1], y=[nice_label(f) for f in feats_sorted[::-1]],
        orientation="h", marker_color=bar_colors[::-1]
    ))
    fig_shap.update_layout(
        title="Explicação da Predição (SHAP) - Paciente em avaliação",
        xaxis_title="Importância SHAP (log-odds)", yaxis_title="Variável",
        height=520, plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_shap, use_container_width=True)
