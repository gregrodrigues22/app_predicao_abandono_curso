# ==============================================================
# Predição de Abandono de Curso  (Streamlit + CatBoost)
# ==============================================================

from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool

# --------------------------------------------------------------
# Config da página
# --------------------------------------------------------------
st.set_page_config(
    page_title="🎓 Predição de Abandono de Curso",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_catboost_fold0.cbm"

ASSETS = BASE_DIR / "assets"
LOGO = (ASSETS / "logo.png") if (ASSETS / "logo.png").exists() else None

def metric_card(title: str, value: str, help: str | None = None) -> None:
    st.metric(title, value, help=help)

# --------------------------------------------------------------
# Variáveis e ordem (EXATAMENTE a do treinamento!)
# --------------------------------------------------------------
FEATURES = [
    "aluno_nascimento_sexo",
    "aluno_nascimento_faixaetaria",
    "aluno_nascimento_raca_descricao",
    "aluno_pessoal_estadocivil_descricao",
    "aluno_profissional_escolaridade_descricao",
    "aluno_profissional_profissao_descricao",
    "aluno_profissional_cnes_prof_sus",
    "aluno_profissional_cnes_profnsus",
    "aluno_acesso_acesso_uf",
    "flag_hist_12m_aband_mesma_cat",
    "status_formulario_inicial",
    "aluno_nascimento_pais",
    "aluno_profissional_cnes_horaoutr",
    "aluno_profissional_cnes_horahosp",
    "aluno_profissional_cnes_hora_amb",
    "matricula_ingresso_inicio_dias",
    "hist_total_cursos",
    "hist_total_trancamentos",
    "hist_total_abandonos",
    "hist_total_conclusoes",
    "hist_12m_total_cursos",
    "hist_12m_trancamentos",
    "hist_12m_abandonos",
    "hist_12m_conclusoes",
    "form_inicial_atual_q01_forma_informacao_sobre_curso",
    "form_inicial_atual_q02_melhorar_desempenho",
    "form_inicial_atual_q02_ampliar_conhecimento",
    "form_inicial_atual_q02_certificado",
    "form_inicial_atual_q02_resolver_problema_real",
    "form_inicial_atual_q02_recomendacao_empregador",
    "form_inicial_atual_q03_nivel_conhecimento",
    "form_inicial_atual_q04_intencionalidade_inicial",
]

# Campos categóricos (exatamente como no treino)
CATEGORICAL_FEATURES = [
    "aluno_nascimento_sexo",
    "aluno_nascimento_faixaetaria",
    "aluno_nascimento_raca_descricao",
    "aluno_pessoal_estadocivil_descricao",
    "aluno_profissional_escolaridade_descricao",
    "aluno_profissional_profissao_descricao",
    "aluno_profissional_cnes_prof_sus",
    "aluno_profissional_cnes_profnsus",
    "aluno_acesso_acesso_uf",
    "flag_hist_12m_aband_mesma_cat",
    "status_formulario_inicial",
    "aluno_nascimento_pais"
]

CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]

# --------------------------------------------------------------
# Carregar modelo
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: Path) -> CatBoostClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path.name}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model

try:
    model = load_model(MODEL_FILE)
except Exception as e:
    st.error(f"Falha ao carregar modelo CatBoost: {e}")
    st.stop()

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_column_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    with st.expander("Predição", expanded=True):
        st.page_link("app.py", label="Predição de Abandono de Curso", icon="📈")

    with st.expander("Explicação", expanded=True):
        st.page_link("pages/explain.py", label="Explicação do Modelo", icon="📙")

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)

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
    """,
    unsafe_allow_html=True
)

# ==============================================================
# FORMULÁRIO – seções e campos
# ==============================================================

# --------- Seção: Dados do(a) aluno(a) ----------
st.subheader("Dados do(a) aluno(a)")

sexo_opts = ["F", "M", "Não identificado"]
faixa_opts = [
    "Jovens adultos (18-29)", "Adultos jovens (30-39)", "Meia-idade inicial (40-49)",
    "Meia-idade avançada (50-59)", "Idosos jovens (60-69)", "Idosos (70-79)",
    "Longevos (80-150)", "Adolescência (15-17)", "Não identificado",
]
raca_opts = ["Branca", "Parda", "Preta", "Amarela", "Indígena", "Não identificado"]
estadocivil_opts = ["Solteiro", "Casado", "União Estável", "Divorciado", "Viúvo", "Não identificado"]

pais_opts = [
    "Brasil", "Portugal", "Angola", "Moçambique", "Cabo Verde",
    "Guiné-Bissau", "São Tomé e Príncipe", "Timor-Leste", "Outro",
]

c1, c2, c3 = st.columns(3)
with c1:
    aluno_nascimento_sexo = st.selectbox("Sexo", sexo_opts, index=0)
    aluno_nascimento_faixaetaria = st.selectbox("Faixa etária", faixa_opts, index=1)
with c2:
    aluno_nascimento_raca_descricao = st.selectbox("Raça/Cor", raca_opts, index=0)
    aluno_pessoal_estadocivil_descricao = st.selectbox("Estado civil", estadocivil_opts, index=0)
with c3:
    aluno_nascimento_pais = st.selectbox("País de nascimento", pais_opts, index=0)
    if aluno_nascimento_pais == "Outro":
        pais_outro = st.text_input("Qual país?")
        if pais_outro.strip():
            aluno_nascimento_pais = pais_outro.strip()

st.markdown("---")

# --------- Seção: Situação profissional ----------
st.subheader("Situação profissional")

# mesma linha: Escolaridade & Profissão
escolaridade_opts = [
    "Ensino Fundamental", "Ensino Médio", "Técnico de Nível Médio", "Graduação",
    "Graduação Tecnológica", "Especialização", "Mestrado Profissional",
    "Mestrado Acadêmico", "Residência Médica", "Residência Multiprofissional",
    "Doutorado", "Não identificado",
]
profissao_opts = [
    "Estudante", "Enfermeiro", "Médico", "Técnico de Enfermagem", "Farmacêutico",
    "Auxiliar de Enfermagem", "Biomédico", "Dentista", "Fisioterapeuta", "Biólogo",
    "Assistente Social", "Nutricionista", "Psicólogo", "Médico Veterinário",
    "Profissionais de Educação Física", "Fonoaudiólogo", "Terapeuta Ocupacional",
    "Agente Comunitário de Saúde", "Outros", "Não identificado",
]
colE, colP = st.columns(2)
with colE:
    aluno_profissional_escolaridade_descricao = st.selectbox("Escolaridade", escolaridade_opts, index=3)
with colP:
    aluno_profissional_profissao_descricao = st.selectbox("Profissão", profissao_opts, index=0)

# mesma linha: SUS & NÃO-SUS
snni = ["Sim", "Não", "Não identificado"]
colSUS, colNSUS = st.columns(2)
with colSUS:
    aluno_profissional_cnes_prof_sus = st.selectbox("Profissional SUS (CNES)", snni, index=1)
with colNSUS:
    aluno_profissional_cnes_profnsus = st.selectbox("Profissional NÃO-SUS (CNES)", snni, index=2)

# horas/dias como SLIDERS
h1, h2, h3 = st.columns(3)
with h1:
    aluno_profissional_cnes_horaoutr = st.slider("Horas OUTR (CNES)", 0, 100, 0)
with h2:
    aluno_profissional_cnes_horahosp = st.slider("Horas HOSP (CNES)", 0, 100, 0)
with h3:
    aluno_profissional_cnes_hora_amb = st.slider("Horas AMB (CNES)", 0, 100, 0)

st.markdown("---")

# --------- Seção: Acesso / UF ----------
st.subheader("Acesso / UF")
uf_opts = [
    "SP", "MG", "PE", "PR", "RJ", "BA", "RS", "CE", "SC", "GO", "ES", "DF", "PA", "MS",
    "PB", "AM", "MA", "SE", "MT", "RO", "RN", "AL", "PI", "TO", "AC", "AP", "RR",
    "Não identificado",
]
aluno_acesso_acesso_uf = st.selectbox("UF de acesso", uf_opts, index=0)

st.markdown("---")

# --------- Seção: Histórico acadêmico ----------
st.subheader("Histórico acadêmico")

c11, c12, c13, c14 = st.columns(4)
with c11:
    hist_total_cursos = st.slider("Total cursos (histórico)", 0, 50, 1)
with c12:
    hist_total_trancamentos = st.slider("Total trancamentos", 0, 50, 0)
with c13:
    hist_total_abandonos = st.slider("Total abandonos", 0, 50, 0)
with c14:
    hist_total_conclusoes = st.slider("Total conclusões", 0, 50, 0)

c15, c16, c17, c18 = st.columns(4)
with c15:
    hist_12m_total_cursos = st.slider("Cursos (12m)", 0, 50, 1)
with c16:
    hist_12m_trancamentos = st.slider("Trancamentos (12m)", 0, 50, 0)
with c17:
    hist_12m_abandonos = st.slider("Abandonos (12m)", 0, 50, 0)
with c18:
    hist_12m_conclusoes = st.slider("Conclusões (12m)", 0, 50, 0)

c19, c20 = st.columns(2)
with c19:
    flag_hist_12m_aband_mesma_cat = st.selectbox("Abandono 12m na mesma categoria?", ["Não", "Sim"], index=0)
with c20:
    matricula_ingresso_inicio_dias = st.slider("Dias desde o ingresso (matrícula)", 0, 365, 30)

st.markdown("---")

# --------- Seção: Formulário inicial ----------
st.subheader("Formulário inicial")

status_formulario_inicial = st.selectbox(
    "Status do formulário inicial",
    ["Respondeu formulário", "Não respondeu"],
    index=0,
)

# Q01 exibido sem numeração e mapeado para o rótulo do treino
if status_formulario_inicial == "Respondeu formulário":
    q01_display_opts = [
        "Pelo portal da UNA-SUS",
        "Pela indicação de outra pessoa (ex.: um colega, amigo,etc.)",
        "Pela Plataforma Arouca",
        "Outro (especifique)",
        "Por uma rede social (Twitter, Facebook, Linkedin, etc.)",
        "Por um material impresso (cartaz, folder, jornal, etc.)",
        "Pelo portal ou blog vinculado ao Ministério da Saúde",
        "Por um site de busca (Google, Yahoo, etc.)",
    ]

    # mapeamento texto -> CÓDIGO numérico usados no treino (float)
    Q01_MAP_TO_CODE = {
        "Pela indicação de outra pessoa (ex.: um colega, amigo,etc.)": 1.0,
        "Pelo portal da UNA-SUS": 2.0,
        # tradicionalmente “Outro” é 3 nos seus questionários
        "Outro (especifique)": 3.0,
        "Pela Plataforma Arouca": 4.0,
        "Por uma rede social (Twitter, Facebook, Linkedin, etc.)": 5.0,
        "Pelo portal ou blog vinculado ao Ministério da Saúde": 6.0,
        "Por um material impresso (cartaz, folder, jornal, etc.)": 7.0,
        "Por um site de busca (Google, Yahoo, etc.)": 10.0,
    }

    if status_formulario_inicial == "Respondeu formulário":
        q01_display_choice = st.selectbox("Q01: Como conheceu o curso?", q01_display_opts, index=0)

        q01_outro_txt = ""
        if q01_display_choice == "Outro (especifique)":
            q01_outro_txt = st.text_input("Descreva (Q01 - Outro):")

        # valor que VAI para o modelo: float
        form_q01_value = float(Q01_MAP_TO_CODE[q01_display_choice])

        # sliders (0–10) já estavam corretos; mantemos como float também
        cQ1, cQ2 = st.columns(2)
        with cQ1:
            q02_ampliar  = st.slider("Q02: ampliar conhecimento (0–10)",      0.0, 10.0, 6.0, 0.5)
            q02_resolver = st.slider("Q02: resolver problema real (0–10)",     0.0, 10.0, 4.0, 0.5)
            q03_nivel    = st.slider("Q03: nível de conhecimento (0–10)",      0.0, 10.0, 5.0, 0.5)
        with cQ2:
            q02_melhorar = st.slider("Q02: melhorar desempenho (0–10)",        0.0, 10.0, 2.0, 0.5)
            q02_cert     = st.slider("Q02: certificado (0–10)",                 0.0, 10.0, 7.0, 0.5)
            q02_recomend = st.slider("Q02: recomendação do empregador (0–10)", 0.0, 10.0, 1.0, 0.5)

        q04_intenc = st.slider("Q04: intencionalidade inicial (0–10)", 0.0, 10.0, 5.0, 0.5)
    else:
        # formulário não respondido → Q01 com um código neutro/“não respondeu” (defina o que usou no treino; se não existir, use 0.0)
        form_q01_value = 0.0
        q02_ampliar = q02_resolver = q03_nivel = q02_melhorar = q02_cert = q02_recomend = q04_intenc = 0.0

st.markdown("---")

# Limiar (opcional)
th_map = {"0.30 (triagem ampla)": 0.30, "0.50 (padrão)": 0.50, "0.70 (mais precisão)": 0.70}
choice = st.radio("Escolha o limiar de decisão", list(th_map.keys()), index=1, horizontal=True)
LIMIAR = th_map[choice]

# Botão de envio
submit = st.button("Calcular probabilidade", type="primary", use_container_width=True)

# --------------------------------------------------------------
# Inferência
# --------------------------------------------------------------
if submit:
    # Q02–Q04: usar diretamente 0–10
    form_q02_melh = float(q02_melhorar)
    form_q02_ampl = float(q02_ampliar)
    form_q02_cert = float(q02_cert)
    form_q02_prob = float(q02_resolver)
    form_q02_reco = float(q02_recomend)
    form_q03      = float(q03_nivel)
    form_q04      = float(q04_intenc)

    row = [
        aluno_nascimento_sexo,
        aluno_nascimento_faixaetaria,
        aluno_nascimento_raca_descricao,
        aluno_pessoal_estadocivil_descricao,
        aluno_profissional_escolaridade_descricao,
        aluno_profissional_profissao_descricao,
        aluno_profissional_cnes_prof_sus,
        aluno_profissional_cnes_profnsus,
        aluno_acesso_acesso_uf,
        flag_hist_12m_aband_mesma_cat,
        status_formulario_inicial,
        aluno_nascimento_pais,
        int(aluno_profissional_cnes_horaoutr),
        int(aluno_profissional_cnes_horahosp),
        int(aluno_profissional_cnes_hora_amb),
        int(matricula_ingresso_inicio_dias),
        int(hist_total_cursos),
        int(hist_total_trancamentos),
        int(hist_total_abandonos),
        int(hist_total_conclusoes),
        int(hist_12m_total_cursos),
        int(hist_12m_trancamentos),
        int(hist_12m_abandonos),
        int(hist_12m_conclusoes),

        # >>> AQUI: Q01 agora vai como FLOAT (código)
        float(form_q01_value),

        # Q02/Q03/Q04 0–10 como float
        float(q02_melhorar),
        float(q02_ampliar),
        float(q02_cert),
        float(q02_resolver),
        float(q02_recomend),
        float(q03_nivel),
        float(q04_intenc),
    ]

    X_pool = Pool(data=[row], cat_features=CAT_IDX)

    prob_abandono = float(model.predict_proba(X_pool)[0, 1])
    prob_nao = 1.0 - prob_abandono

    # ---------------------- Exibição ----------------------
    st.subheader("Resultado")
    classe = "Alto risco de abandono" if prob_abandono >= LIMIAR else "Baixo risco de abandono"

    colA, colB = st.columns([1, 2])
    with colA:
        metric_card("Classe prevista", classe, help=f"Limiar atual: {LIMIAR:.2f}")
        st.write(f"**Prob. Abandono (1):** {prob_abandono:.1%}")
        st.write(f"**Prob. Não-Abandono (0):** {prob_nao:.1%}")

    with colB:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Não-Abandono", "Abandono"],
            y=[prob_nao, prob_abandono],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{prob_nao:.1%}", f"{prob_abandono:.1%}"],
            textposition="auto",
        ))
        fig.add_hline(
            y=LIMIAR,
            line_dash="dash",
            line_color="#222",
            annotation_text=f"Limiar {LIMIAR:.2f}",
            annotation_position="top left",
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Probabilidade"),
            xaxis=dict(title="Classe"),
            bargap=0.25,
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dados enviados (na ordem do treino)"):
        st.json({k: v for k, v in zip(FEATURES, row)})

    st.caption("Aviso: apoio à decisão; não substitui julgamento acadêmico.")