from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_fscore_support,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score
)
from sklearn.calibration import calibration_curve

# --------------------------------------------------------------
# Config & paths
# --------------------------------------------------------------
st.set_page_config(page_title="🧠 Explicação do Modelo", page_icon="🧠", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "model.cbm"
META_FILE  = BASE_DIR / "metadata.json"

OOF_FILE = BASE_DIR / "oof_all.parquet"
if not OOF_FILE.exists():
    alt = BASE_DIR / "off_all.parquet"
    if alt.exists():
        OOF_FILE = alt

ASSETS = BASE_DIR / "assets"
LOGO = next((p for p in ["logo.png", "logo.jpg", "logo.webp"] if (ASSETS / p).exists()), None)

# Esconde a navegação multipágina padrão (você já tem um menu)
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"], [data-testid="stSidebarHeader"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def metric_card(title: str, value: str, help: str | None = None):
    st.metric(title, value, help=help)

# --------------------------------------------------------------
# Sidebar (menu próprio)
# --------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(ASSETS / LOGO), use_column_width=True)
    st.header("Menu")
    st.page_link("app.py", label="Predição", icon="📈")
    st.page_link("pages/explain.py", label="Explicação", icon="📙")
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)

# --------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(90deg,#0f4c81,#002b55);
                padding:28px;border-radius:14px;margin-bottom:22px'>
      <h1 style='color:#fff;margin:0'>🧠 Explicação do Modelo</h1>
      <p style='color:#dfe9f3;margin:6px 0 0 0'>
        Desempenho (ROC, PR, matriz, métricas) e interpretação (FI, SHAP) do classificador CatBoost
        para risco de abandono/conclusão.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------
# Texto explicativo (seções)
# --------------------------------------------------------------
def render_explanatory_sections():
    st.header("Contexto e motivação")
    st.markdown("""
A evasão reduz o impacto das ações formativas e encarece a política pública.  
Aqui estimamos **probabilidade de conclusão (classe 1)** e, complementarmente, **probabilidade de abandono (1 − prob_conclusão)**, 
para **apoiar ações de retenção** (mensagens, tutoria, flexibilização de prazos etc.).
""")

    st.header("Substitutos e abordagens tradicionais")
    st.markdown("""
Regras fixas e indicadores isolados (ex.: dias desde o ingresso, histórico de conclusões) ajudam,
mas não capturam **relações não-lineares** nem **interações** entre variáveis, gerando muitos falsos positivos/negativos.
""")

    st.header("Solução proposta")
    st.markdown("""
Treinamos um **CatBoost Classifier** com **9 variáveis** selecionadas por desempenho/estabilidade.
O modelo retorna **probabilidade de Conclusão (1)**.  
**Features finais** (ordem do treino):  
- `status_formulario_inicial`  
- `matricula_ingresso_inicio_dias`  
- `aluno_profissional_profissao_descricao`  
- `hist_12m_conclusoes`  
- `hist_total_conclusoes`  
- `hist_total_abandonos`  
- `form_inicial_atual_q03_nivel_conhecimento`  
- `aluno_acesso_acesso_uf`  
- `aluno_profissional_escolaridade_descricao`
""")

    st.header("Metodologia (resumo)")
    st.markdown("""
1) **Validação cruzada** com OOF para estimar generalização.  
2) **Seleção de features** (estabilidade + desempenho).  
3) **Treino** do CatBoost com categóricas nativas.  
4) **Avaliação OOF**: ROC/AUC, PR/AP, matriz de confusão e relatório no **limiar escolhido**.  
5) **Interpretação**: Importância global (FI) e **valores SHAP** (impacto por variável).
""")

    st.header("Como interpretar os gráficos")
    st.markdown("""
- **ROC/AUC**: separação de **Conclusão (1)** vs **Abandono (0)** em todos os limiares.  
- **PR/AP**: precisão vs recall para a classe 1 — útil quando desbalanceada.  
- **Matriz de Confusão**: no **limiar atual**, mostra acertos/erros; ajuste o limiar à sua capacidade de intervenção.  
- **Relatório**: precisão/recall/F1 por classe.  
- **FI**: ranking global de variáveis pelo ganho médio.  
- **SHAP**:  
  - valores **positivos** → empurram para **Conclusão (1)**;  
  - valores **negativos** → empurram para **Abandono (0)**;  
  - o **|SHAP| médio** indica o **peso** de cada variável no conjunto analisado.
""")

    st.header("Boas práticas e limitações")
    st.markdown("""
- Defina **limiares operacionais** (ex.: alto risco se `prob_conclusao < 0.45`) conforme **custo/benefício** das ações.  
- Monitore **métricas por subgrupos** (UF, profissão, escolaridade) para equidade.  
- **Recalibre/re-treine** periodicamente (sazonalidade, mudança de público).  
- O desempenho aqui é **OOF**; avalie também em produção. Resultados são **probabilísticos**, **não determinísticos**.
""")

render_explanatory_sections()

# --------------------------------------------------------------
# Carregar artefatos
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: Path) -> CatBoostClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    m = CatBoostClassifier()
    m.load_model(str(path))
    return m

@st.cache_resource(show_spinner=False)
def load_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError("metadata.json não encontrado.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    features = meta.get("selected_features") or meta.get("feature_order") or []
    cat_feats = meta.get("cat_features") or []
    params = meta.get("params", {})
    created = meta.get("created_at", "—")
    catboost_ver = meta.get("catboost_version", "—")
    return features, cat_feats, params, dict(created_at=created, catboost=catboost_ver)

@st.cache_data(show_spinner=False)
def load_oof(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} não encontrado na raiz do projeto.")
    return pd.read_parquet(path)  # engine pyarrow/fastparquet conforme requirements

# Load
try:
    model = load_model(MODEL_FILE)
    FEATURES, CAT_FEATS, CB_PARAMS, META_INFO = load_metadata(META_FILE)
    df_oof = load_oof(OOF_FILE)
    st.success("✅ Artefatos carregados (modelo, metadata e OOF).")
except Exception as e:
    st.error(f"Falha ao carregar artefatos: {e}")
    st.stop()

# Metainformações
ic1, ic2, ic3 = st.columns(3)
ic1.metric("Criado em", str(META_INFO.get("created_at", "—")))
ic2.metric("CatBoost", str(META_INFO.get("catboost", "—")))
ic3.metric("#features", str(len(FEATURES)))

with st.expander("Parâmetros e Features do treino"):
    c1, c2 = st.columns(2)
    with c1:
        st.json(CB_PARAMS)
    with c2:
        st.write("**Features (ordem):**", FEATURES)
        st.write("**Categóricas:**", CAT_FEATS)

# --------------------------------------------------------------
# Desempenho OOF: ROC/AUC + PR/AP + Confusão + Relatório
# --------------------------------------------------------------
st.subheader("Desempenho em validação (OOF)")

# Colunas mínimas
col_true = "y_true" if "y_true" in df_oof.columns else "target"
col_prob = "y_pred_proba" if "y_pred_proba" in df_oof.columns else None
for cand in ["pred_1", "proba", "p_class1"]:
    if col_prob is None and cand in df_oof.columns:
        col_prob = cand

if col_true not in df_oof.columns or col_prob not in df_oof.columns:
    st.error("O OOF precisa conter, no mínimo, as colunas **y_true** e **y_pred_proba**.")
    st.stop()

y_true = df_oof[col_true].astype(int).values
proba  = df_oof[col_prob].astype(float).values

# ROC/AUC
auc = roc_auc_score(y_true, proba)
fpr, tpr, _ = roc_curve(y_true, proba)

# PR/AP
ap = average_precision_score(y_true, proba)
pr, rc, _ = precision_recall_curve(y_true, proba)

c1, c2 = st.columns([1.25, 1])
with c1:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Aleatório", line=dict(dash="dash")))
    fig_roc.update_layout(template="plotly_white", height=360,
                          xaxis_title="FPR (falsos positivos)",
                          yaxis_title="TPR (verdadeiros positivos)")
    st.plotly_chart(fig_roc, use_container_width=True)

with c2:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig_pr.update_layout(template="plotly_white", height=360,
                         xaxis_title="Recall (classe 1)",
                         yaxis_title="Precision (classe 1)")
    st.plotly_chart(fig_pr, use_container_width=True)

# Escolha de limiar
st.markdown("### Limiar de decisão")
lc1, lc2, lc3 = st.columns([1, 1, 1])

with lc1:
    th_cur = st.slider("Limiar manual", 0.0, 1.0, 0.50, 0.01, key="th_cur")

with lc2:
    opt_choice = st.selectbox(
        "Otimizar limiar para",
        ["Youden (TPR - FPR)", "F1", "F0.5", "F2"], index=0
    )

def best_threshold(y, p, how="Youden (TPR - FPR)"):
    fpr, tpr, thr = roc_curve(y, p)
    if how.startswith("Youden"):
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    beta_map = {"F1":1.0, "F0.5":0.5, "F2":2.0}
    if how in beta_map:
        grid = np.linspace(0.01, 0.99, 99)
        scores = [fbeta_score(y, (p>=t).astype(int), beta=beta_map[how]) for t in grid]
        return float(grid[int(np.argmax(scores))])
    return 0.5

auto_th = best_threshold(y_true, proba, opt_choice)

with lc3:
    use_auto = st.checkbox(f"Usar limiar sugerido ({opt_choice}: {auto_th:.3f})", value=False)

th = auto_th if use_auto else th_cur
st.caption(f"Limiar efetivo: **{th:.3f}**")

# Predições no limiar
y_hat = (proba >= th).astype(int)
cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = (tp + tn) / cm.sum()

mc1, mc2 = st.columns(2)
with mc1:
    metric_card("AUC", f"{auc:.3f}")
    metric_card("AP (PR)", f"{ap:.3f}")
with mc2:
    metric_card("Acurácia", f"{acc:.3f}")
    metric_card("Prevalência classe 1", f"{(y_true.mean()):.3f}")

# Matriz de confusão
st.markdown("### Matriz de Confusão (no limiar atual)")
norm = st.checkbox("Normalizar por linha (%)", value=True)
cm_plot = cm.astype(float)
if norm:
    cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
fig_cm = go.Figure(go.Heatmap(
    z=cm_plot,
    x=["Pred 0 (Abandono)", "Pred 1 (Conclusão)"],
    y=["Real 0 (Abandono)", "Real 1 (Conclusão)"],
    text=np.round(cm_plot * (100 if norm else 1), 1),
    texttemplate="%{text}",
    colorscale="Blues", showscale=False
))
fig_cm.update_layout(template="plotly_white", height=320)
st.plotly_chart(fig_cm, use_container_width=True)

# Relatório de classificação
st.markdown("### Relatório de Classificação (no limiar atual)")
st.code(classification_report(y_true, y_hat, digits=3), language="text")

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_hat, labels=[0, 1], zero_division=0
)
st.table(pd.DataFrame({
    "classe": ["0 (Abandono)", "1 (Conclusão)"],
    "precisão": np.round(prec, 3),
    "recall":   np.round(rec, 3),
    "F1":       np.round(f1, 3),
}))

st.markdown("---")

# --------------------------------------------------------------
# Análises adicionais: Decis/Lift & Calibração
# --------------------------------------------------------------
st.subheader("Análises adicionais")

# Decis / lift
dec = pd.DataFrame({"y_true": y_true, "proba": proba}).sort_values("proba", ascending=False)
dec["decil"] = pd.qcut(dec["proba"], 10, labels=False, duplicates="drop") + 1
tab = dec.groupby("decil").agg(
    n=("y_true", "size"),
    concl=("y_true", "sum"),
    p_media=("proba", "mean")
).sort_index(ascending=False)
tab["taxa_concl"] = tab["concl"] / tab["n"]
tab["lift"] = tab["taxa_concl"] / dec["y_true"].mean()
st.markdown("#### Análise por decil (top-10% até bottom-10%)")
st.dataframe(tab.style.format({"p_media":"{:.3f}", "taxa_concl":"{:.3f}", "lift":"{:.2f}"}), use_container_width=True)

# Calibração (reliability)
prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibração"))
fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfeito", line=dict(dash="dash")))
fig_cal.update_layout(template="plotly_white", height=320,
                      xaxis_title="Prob. prevista (média no bin)",
                      yaxis_title="Taxa real de conclusão")
st.plotly_chart(fig_cal, use_container_width=True)

# --------------------------------------------------------------
# Interpretação: FI + SHAP (somente se o OOF contiver as 9 features)
# --------------------------------------------------------------
st.subheader("Interpretação (Importância de Features e SHAP)")

has_all_feats = all(c in df_oof.columns for c in FEATURES)
if has_all_feats:
    Xint = df_oof[FEATURES].copy()
    for c in CAT_FEATS:
        Xint[c] = Xint[c].astype("string").fillna("Não identificado")
    CAT_IDX = [FEATURES.index(c) for c in CAT_FEATS]
    pool = Pool(Xint, cat_features=CAT_IDX)

    # FI (PredictionValuesChange)
    fi = model.get_feature_importance(pool, type="FeatureImportance")
    df_fi = pd.DataFrame({"feature": FEATURES, "importance": fi}).sort_values("importance", ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=df_fi["importance"], y=df_fi["feature"], orientation="h",
        text=df_fi["importance"].round(2), textposition="auto"
    ))
    fig_fi.update_layout(template="plotly_white", height=380, xaxis_title="Importância (ganho/perda)")
    st.plotly_chart(fig_fi, use_container_width=True)

    # SHAP nativo do CatBoost
    with st.spinner("Calculando SHAP..."):
        shap_vals = np.array(model.get_feature_importance(pool, type="ShapValues"))
        shap_matrix = shap_vals[:, :-1]  # última coluna é base value

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    df_shap = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=True
    )
    fig_shap = go.Figure(go.Bar(
        x=df_shap["mean_abs_shap"], y=df_shap["feature"], orientation="h",
        text=df_shap["mean_abs_shap"].round(3), textposition="auto"
    ))
    fig_shap.update_layout(template="plotly_white", height=380, xaxis_title="|SHAP| médio (impacto absoluto)")
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("#### Explicação local (Top-10 contribuições)")
    idx_local = st.number_input("Índice na amostra", 0, shap_matrix.shape[0]-1, 0, 1)
    row_shap = shap_matrix[int(idx_local)]
    order = np.argsort(np.abs(row_shap))[::-1][:10]
    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(
        x=row_shap[order], y=[FEATURES[i] for i in order], orientation="h",
        marker_color=np.where(row_shap[order] >= 0, "#d62728", "#1f77b4"),
        text=[f"{v:+.3f}" for v in row_shap[order]], textposition="auto"
    ))
    fig_w.add_vline(x=0, line_dash="dash")
    fig_w.update_layout(template="plotly_white", height=420,
                        xaxis_title="Contribuição SHAP (log-odds)")
    st.plotly_chart(fig_w, use_container_width=True)

else:
    st.warning(
        "O **oof_all.parquet** atual não contém as 9 features selecionadas. "
        "Sem as colunas de entrada, não é possível calcular **FI/SHAP**. "
        "Para habilitar essa seção, gere um OOF que inclua as features "
        "além de `y_true` e `y_pred_proba`."
    )

# --------------------------------------------------------------
# Análise por subgrupos & Download
# --------------------------------------------------------------
st.subheader("Equidade/operacional: métricas por subgrupos")

# Sugestões de colunas categóricas possíveis que podem estar no OOF enriquecido
candidate_groups = [
    "aluno_acesso_acesso_uf",
    "aluno_profissional_profissao_descricao",
    "aluno_profissional_escolaridade_descricao",
    "status_formulario_inicial"
]
choices = [c for c in candidate_groups if c in df_oof.columns]
if choices:
    subcol = st.selectbox("Escolha a variável categórica", choices)
    grp = df_oof.copy()
    grp["y_hat"] = (proba >= th).astype(int)

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "prec": precision_score(g[col_true], g["y_hat"], zero_division=0),
            "rec":  recall_score(g[col_true], g["y_hat"], zero_division=0),
            "f1":   f1_score(g[col_true], g["y_hat"], zero_division=0),
            "auc":  roc_auc_score(g[col_true], g[col_prob]) if g[col_true].nunique() == 2 else np.nan
        })
    st.dataframe(
        grp.groupby(subcol, dropna=False).apply(_agg).sort_values("n", ascending=False),
        use_container_width=True
    )
else:
    st.info("Adicione colunas categóricas no OOF para análise por subgrupos (ex.: UF, profissão, escolaridade).")