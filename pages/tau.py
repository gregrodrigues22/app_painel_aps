# ---------------------------------------------------------------
# TAU – Upload + Validação + Gráficos
# ---------------------------------------------------------------
import io
import re
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from src.plots import bar_yoy_trend, fig_ativos_por_mes, tau_trimestral_acumulado_por_dia, heatmap_combos_por_trimestre_colscale
from src.plots import meta_trimestral_acumulada

# ---------------------------------------------------------------
# Config da página
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="📊 Public Health Analytics — TAU")

# ---------------- Helpers para assets ----------------
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR.parent / "assets"   # este arquivo está em /pages

def first_existing(*relative_paths: str) -> Path | None:
    for rel in relative_paths:
        p = ASSETS / rel
        if p.exists():
            return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# ---------------- Cabeçalho ----------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Ana Health — Engajamento</h1>
        <p style='color: white;'>Explore os painéis para tomada de decisão baseada em dados.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Oculta o menu multipage padrão (mantemos nosso menu lateral)
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- navegação segura ---
def safe_page_link(path: str, label: str, icon: str | None = None):
    try:
        if (APP_DIR.parent / path).exists() or (APP_DIR / path).exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página não disponível neste app.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

# ---------------- Sidebar ----------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.header("Menu")
    safe_page_link("app.py", label="Voltar ao início", icon="↩️")
    st.caption("Atalhos")
    safe_page_link("pages/relatorio_gestor.py", label="Relatório Gestor", icon="🏠")
    safe_page_link("pages/operacao.py", label="Operação", icon="🏨")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Conecte-se")
    st.markdown("""
- 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
- ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
- 📸 [Instagram](https://www.instagram.com/patients2python/)
- 🌐 [Site](https://patients2python.com.br/)
- 🐙 [GitHub](https://github.com/gregrodrigues22)
- 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
- 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
- 🎓 [Escola](https://app.patients2python.com.br/browse)
    """, unsafe_allow_html=True)

# =========================
# Helpers de dados
# =========================
EXPECTED_SCHEMA = {
    # coluna           (tipo esperado, descrição/valid.)
    "id_pessoa":      ("string",  "Identificador (texto)"),
    "sexo":           ("int",     "0/1 (ou 1/2) — será normalizado para 0/1"),
    "faixa_etaria":   ("int",     "inteiro (código de faixa)"),
    "higido":         ("int01",   "0/1"),
    "multicomorbido": ("int01",   "0/1"),
    "diabetes":       ("int01",   "0/1"),
    "dislipidemia":   ("int01",   "0/1"),
    "hipertensao":    ("int01",   "0/1"),
    "obesidade":      ("int01",   "0/1"),
    "saude_mental":   ("int01",   "0/1"),
    "data":           ("date",    "data no formato AAAA-MM-DD"),
    "mes_atendimento":("date",    "primeiro dia do mês AAAA-MM-DD"),
    "tipo":           ("string",  "texto (ex.: atendimento)")
}

COND_COLS = ["higido","multicomorbido","diabetes","dislipidemia","hipertensao","obesidade","saude_mental"]

@st.cache_data(show_spinner=False)
def _read_csv_smart(file, force_sep: str | None = None, dtype_map: dict | None = None) -> pd.DataFrame:
    dtype_map = dtype_map or {}
    if force_sep:
        return pd.read_csv(file, sep=force_sep, dtype=dtype_map)
    head = file.getvalue().splitlines()[0].decode("utf-8", errors="ignore")
    guess = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.BytesIO(file.getvalue()), sep=guess, dtype=dtype_map)

def schema_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_null": [df[c].isna().sum() for c in df.columns],
        "exemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })

def validate_and_clean(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tenta **coagir** os tipos esperados e devolve:
      - df_clean: dataframe já coerido
      - report  : tabela de validação por coluna
    """
    df = df_in.copy()
    report_rows = []

    # normalizações úteis antes dos casts
    if "id_pessoa" in df.columns:
        df["id_pessoa"] = df["id_pessoa"].astype("string")

    if "sexo" in df.columns:
        # aceita {1,2} ou {0,1}; mapeia 2->0 para virar binário 0/1 (ajuste conforme sua regra)
        s = pd.to_numeric(df["sexo"], errors="coerce").astype("Int64")
        s = s.replace({2:0})
        df["sexo"] = s.fillna(0).astype(int)

    # coerções por schema
    for col, (expected, _) in EXPECTED_SCHEMA.items():
        status = "ok"
        details = ""
        present = col in df.columns

        if not present:
            status = "faltando"
            report_rows.append((col, expected, "-", status, "Coluna ausente no CSV"))
            continue

        try:
            if expected == "string":
                df[col] = df[col].astype("string")
            elif expected == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").fillna(0).astype(int)
            elif expected == "int01":
                s = pd.to_numeric(df[col], errors="coerce").fillna(0)
                # qualq. valor >0 vira 1
                df[col] = (s > 0).astype(int)
            elif expected == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                # fallback
                df[col] = df[col]
        except Exception as e:
            status = "erro"
            details = f"Falha ao converter para {expected}: {e}"

        # checagens específicas
        if expected == "int01" and present:
            invalid = (~df[col].isin([0,1])).sum()
            if invalid > 0:
                status = "alerta"
                details = f"{invalid} valores não binários (ajustados para 0/1)."

        if expected == "date" and present:
            n_na = df[col].isna().sum()
            if n_na > 0:
                status = "alerta" if status == "ok" else status
                details = (details + " " if details else "") + f"{n_na} datas inválidas (NaT)."

        report_rows.append((col, expected, str(df[col].dtype), status, details.strip()))

    # colunas extras
    extras = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    for c in extras:
        report_rows.append((c, "-", str(df[c].dtype), "extra", "Coluna não esperada (mantida)."))

    report = pd.DataFrame(report_rows, columns=["coluna","esperado","dtype_final","status","detalhes"])
    return df, report

# =========================
# Abas
# =========================
tab1, tab2 = st.tabs(["📥 Importar & Validar", "📈 Gráficos"])

# --------- Aba 1
with tab1:
    st.subheader("📤 Carregue seu CSV")
    st.write("Formato esperado e tipos por coluna:")

    sch = pd.DataFrame([
        {"coluna": c, "tipo_esperado": t[0], "descrição": t[1]}
        for c, t in EXPECTED_SCHEMA.items()
    ])
    st.dataframe(sch, use_container_width=True, hide_index=True)

    uploaded = st.file_uploader(
        "Arraste e solte o arquivo aqui, ou clique para selecionar",
        type=["csv", "txt"],
        accept_multiple_files=False,
        help="Tamanho máximo por arquivo: ~200MB"
    )

    sep_override = st.radio(
        "Separador",
        options=["Detectar automaticamente", "Vírgula (,)", "Ponto e vírgula (;)"],
        horizontal=True
    )
    force_sep = None
    if sep_override == "Vírgula (,)":
        force_sep = ","
    elif sep_override == "Ponto e vírgula (;)":
        force_sep = ";"

    if uploaded is not None:
        with st.spinner("⏳ Carregando e validando..."):
            try:
                df_raw = _read_csv_smart(uploaded, force_sep=force_sep, dtype_map={"id_pessoa": "string"})
                df, rep = validate_and_clean(df_raw)

                st.success(f"✅ Arquivo carregado: **{uploaded.name}** — {df.shape[0]:,} linhas × {df.shape[1]} colunas".replace(",", "."))

                c1, c2 = st.columns([1,1])
                with c1:
                    st.markdown("**Schema detectado**")
                    st.dataframe(schema_df(df), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Prévia (10 linhas)**")
                    st.dataframe(df.head(10), use_container_width=True)

                st.markdown("### ✅ Validação por coluna")
                st.dataframe(rep, use_container_width=True, hide_index=True)

                st.session_state["df_tau"] = df

            except Exception as e:
                st.error(f"❌ Erro ao processar: {e}")
    else:
        st.info("Envie um arquivo para iniciar.")

# =========================
# ABA 2 — Analytics (Filtros → Grandes Números → Gráficos)
# =========================
with tab2:
    st.subheader("📊 Analytics")

    if "df_tau" not in st.session_state:
        st.warning("Nenhum dado validado disponível. Vá à aba **Importar & Validar** e carregue o CSV.")
        st.stop()

    # ---------- PREP BÁSICO ----------

    df = st.session_state["df_tau"].copy()
    date_col = "mes_atendimento" if "mes_atendimento" in df.columns else ("data" if "data" in df.columns else None)
    if not date_col:
        st.warning("Não encontrei coluna de data (`mes_atendimento` ou `data`).")
        st.stop()

    # Datas normalizadas
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_year"] = df["_dt"].dt.year
    df["_month_period"] = df["_dt"].dt.to_period("M")
    df["_quarter"] = df["_dt"].dt.quarter
    df["_yq_lbl"] = (
    df["_dt"].dt.year.astype(str) + "Q" +
    df["_dt"].dt.quarter.astype(str).str.zfill(2))

    # tipificação leve para 'tipo' (interações)
    import unicodedata
    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
        return s
    if "tipo" in df.columns:
        df["_tipo_norm"] = df["tipo"].map(_norm)
    else:
        # série vazia, mantém compatibilidade com os filtros/contagens
        df["_tipo_norm"] = pd.Series("", index=df.index)

    INTER_TIPOS = {"mensagem","ligacao","atendimento"}
    INTER_TIPOS_MS = {"mensagem"}
    INTER_TIPOS_LG = {"ligacao"}
    INTER_TIPOS_AT = {"atendimento"}

    COND_COLS = [c for c in ["higido","multicomorbido","diabetes","dislipidemia","hipertensao","obesidade","saude_mental"] if c in df.columns]

    # ------------------------------
    # Sessão de Filtros
    # ------------------------------
    st.info("🧪 Sessão de Filtros: Use os controles abaixo para refinar os resultados.")
    
    with st.container(border=True):

        # Linha 1 — Sexo | Faixa Etária
        c1, c2 = st.columns(2)
        sexo_opts = ["(Todos)"] + sorted(map(str, df["sexo"].dropna().unique())) if "sexo" in df.columns else ["(N/A)"]
        sexo_sel = c1.selectbox("Sexo", sexo_opts, index=0)

        fe_opts = ["(Todos)"] + sorted(map(str, df["faixa_etaria"].dropna().unique())) if "faixa_etaria" in df.columns else ["(N/A)"]
        fe_sel  = c2.selectbox("Faixa Etária", fe_opts, index=0)

        # Linha 2 — checkboxes das condições
        st.markdown("**Condições**")
        cA, cB, cC, cD, cE, cF, cG = st.columns(7)
        chk_alguma = cA.checkbox("Alguma condição", value=False, help="Filtra quem tem pelo menos 1 condição verdadeira")
        chk_multi  = cB.checkbox("Multicomorbido", value=False) if "multicomorbido" in df.columns else (False)
        chk_diab   = cC.checkbox("Diabetes", value=False) if "diabetes" in df.columns else (False)
        chk_disl   = cD.checkbox("Dislidepidemia", value=False) if "dislipidemia" in df.columns else (False)
        chk_hip   = cE.checkbox("Hipertensão", value=False) if "hipertensao" in df.columns else (False)
        chk_obes  = cF.checkbox("Obesidade", value=False) if "obesidade" in df.columns else (False)
        chk_sm    = cG.checkbox("Saúde Mental", value=False) if "saude_mental" in df.columns else (False)

        # Linha 3 — Período (ano)
        cY = st.columns([1])[0]
        minY, maxY = int(df["_year"].min()), int(df["_year"].max())
        yr1, yr2 = cY.slider("Período (ano)", min_value=minY, max_value=maxY, value=(minY, maxY))

        # Linha 4 — Trimestre (YYYYQQ)
        colQ, _ = st.columns([1, 1])   # <- desempacota!
        yq_all = sorted(df["_yq_lbl"].dropna().unique().tolist())
        if not yq_all:
            yq_all = []
        yq_sel = colQ.multiselect(
            "Trimestre (YYYYQQ)", yq_all, default=yq_all,
            help="Filtra por trimestres. Combine com o intervalo de anos ao lado."
        )

    # ------ aplica filtros de seleção ------
    mask = (df["_year"].between(yr1, yr2)) & (df["_yq_lbl"].isin(yq_sel))
    if sexo_sel != "(Todos)" and "sexo" in df.columns:
        mask &= (df["sexo"].astype(str) == sexo_sel)
    if fe_sel != "(Todos)" and "faixa_etaria" in df.columns:
        mask &= (df["faixa_etaria"].astype(str) == fe_sel)

    # condições
    if chk_alguma and COND_COLS:
        mask &= df[COND_COLS].astype(bool).any(axis=1)
    if chk_multi and "multicomorbido" in df.columns:
        mask &= df["multicomorbido"].astype(bool)
    if chk_diab and "diabetes" in df.columns:
        mask &= df["diabetes"].astype(bool)
    if chk_disl and "dislipidemia" in df.columns:
        mask &= df["dislipidemia"].astype(bool)
    if chk_hip and "hipertensao" in df.columns:
        mask &= df["hipertensao"].astype(bool)
    if chk_obes and "obesidade" in df.columns:
        mask &= df["obesidade"].astype(bool)
    if chk_sm and "saude_mental" in df.columns:
        mask &= df["saude_mental"].astype(bool)

    dfF = df.loc[mask].copy()

    st.caption("Os gráficos e números abaixo respeitam os filtros aplicados.")

    # ------------------------------
    # Sessão de Grandes Números
    # ------------------------------
    st.info("📌 Sessão de Grandes Números: visão rápida com os filtros aplicados.")

    with st.container(border=True):

        # ========= TOPO (4 KPIs) =========
        c1, c2, c3, c4 = st.columns(4)

        pessoas_dist = (dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF))

        if not dfF.empty:
            last_month = dfF["_month_period"].max()
            pessoas_ativas = dfF.loc[dfF["_month_period"] == last_month, "id_pessoa"].nunique()
        else:
            pessoas_ativas = 0

        meses_dist    = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        media_pessoa_mes = float(pessoas_dist / max(1, meses_dist))
        quarters_dist = int(dfF["_yq_lbl"].nunique()) if not dfF.empty else 0
        media_pessoa_quarters = float(pessoas_dist / max(1, quarters_dist))

        def fmt_br(x, nd=2):
            s = f"{x:,.{nd}f}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")

        c1.metric("Pessoas distintas", f"{pessoas_dist:,}".replace(",", "."))
        c2.metric("Pessoas ativas (último período)", f"{pessoas_ativas:,}".replace(",", "."))
        c3.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_mes))
        c4.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_quarters))

        st.markdown("---")

        # ========= TOPO (4 KPIs) =========
        c5, c6, c7, c8 = st.columns(4)

        pessoas_dist = (dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF))

        if not dfF.empty:
            last_month = dfF["_month_period"].max()
            pessoas_ativas = dfF.loc[dfF["_month_period"] == last_month, "id_pessoa"].nunique()
        else:
            pessoas_ativas = 0

        df_inter = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS)]
        num_inter = int(len(df_inter))
        media_inter = float(num_inter / max(1, pessoas_dist))
        df_mensagens = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_MS)]
        num_mensagens = int(len(df_mensagens))
        media_mensagens = float(num_mensagens / max(1, pessoas_dist))

        c5.metric("Número de interações Totais", f"{num_inter:,}".replace(",", "."))
        c6.metric("Média de interações Totais por pessoa", f"{media_inter:,.2f}".replace(",", "."))      
        c7.metric("Número de interações Mensagens", f"{num_mensagens:,}".replace(",", "."))
        c8.metric("Média de interações Mensagens por pessoa", f"{media_mensagens:,.2f}".replace(",", "."))      

        # ========= TOPO (4 KPIs) =========
        c9, c10, c11, c12 = st.columns(4)

        pessoas_dist = (dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF))

        if not dfF.empty:
            last_month = dfF["_month_period"].max()
            pessoas_ativas = dfF.loc[dfF["_month_period"] == last_month, "id_pessoa"].nunique()
        else:
            pessoas_ativas = 0

        df_agendamentos = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_AT)]
        num_agendamentos = int(len(df_agendamentos))
        media_agendamentos = float(num_agendamentos / max(1, pessoas_dist))
        df_ligacoes = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_LG)]
        num_ligacoes = int(len(df_ligacoes))
        media_ligacoes = float(num_ligacoes / max(1, pessoas_dist))

        c9.metric("Número de interações Agendamento", f"{num_agendamentos:,}".replace(",", "."))
        c10.metric("Média de interações Agendamento por pessoa", f"{media_agendamentos:,.2f}".replace(",", "."))      
        c11.metric("Número de interações Ligações", f"{num_ligacoes:,}".replace(",", "."))
        c12.metric("Média de interações Ligações por pessoa", f"{media_ligacoes:,.2f}".replace(",", "."))         

        st.markdown("---")

        # ========= CONDIÇÕES (4 + 3) =========
        if "id_pessoa" in dfF.columns and COND_COLS:
            base_pessoa = dfF.groupby("id_pessoa")[COND_COLS].max(numeric_only=True)
            n_alguma = int(base_pessoa.any(axis=1).sum())
            n_multi  = int(base_pessoa.get("multicomorbido", pd.Series(False, index=base_pessoa.index)).sum())
            n_diab   = int(base_pessoa.get("diabetes", pd.Series(False, index=base_pessoa.index)).sum())
            n_disl   = int(base_pessoa.get("dislipidemia", pd.Series(False, index=base_pessoa.index)).sum())
            n_hip    = int(base_pessoa.get("hipertensao", pd.Series(False, index=base_pessoa.index)).sum())
            n_obes   = int(base_pessoa.get("obesidade", pd.Series(False, index=base_pessoa.index)).sum())
            n_sm     = int(base_pessoa.get("saude_mental", pd.Series(False, index=base_pessoa.index)).sum())
        else:
            n_alguma = n_multi = n_diab = n_disl = n_hip = n_obes = n_sm = 0

        # Linha 1 (4 KPIs de condição)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Com alguma condição", f"{n_alguma:,}".replace(",", "."))
        k2.metric("Multicomorbidos",      f"{n_multi:,}".replace(",", "."))
        k3.metric("Diabéticos",           f"{n_diab:,}".replace(",", "."))
        k4.metric("Dislidepidêmicos",     f"{n_disl:,}".replace(",", "."))

        # Linha 2 (3 KPIs de condição)
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Hipertensos",          f"{n_hip:,}".replace(",", "."))
        k6.metric("Obesos",               f"{n_obes:,}".replace(",", "."))
        k7.metric("Pessoas saúde mental", f"{n_sm:,}".replace(",", "."))

        st.markdown("---")

        # ========= TEMPO (3 KPIs) =========
        t1, t2, t3, t4 = st.columns(4)
        meses_dist    = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        anos_dist     = int(dfF["_year"].nunique())          if not dfF.empty else 0
        quarters_dist = int(dfF["_yq_lbl"].nunique())        if not dfF.empty else 0

        t1.metric("Meses distintos",    f"{meses_dist:,}".replace(",", "."))
        t2.metric("Anos distintos",     f"{anos_dist:,}".replace(",", "."))
        t3.metric("Quarters distintos", f"{quarters_dist:,}".replace(",", "."))

    # ------------------------------
    # Sessão de Gráficos
    # ------------------------------
    st.info("📈 Sessão de Gráficos: visuais para responder às perguntas segundo os filtros aplicados.")

    with st.expander("Qual número de pessoas ativas por mês?", expanded=True):
        fig = fig_ativos_por_mes(
            dfF,                       # seu dataframe filtrado
            date_col="_dt",            # ou "mes_atendimento" se você quiser usar essa coluna
            id_col="id_pessoa",
            trend_alpha=0.10,          # mais baixo = mais suave
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("Qual é o nível engajamento por trimestre?", expanded=True):
        date_col_for_tau = (
            "_dt" if "_dt" in dfF.columns else
            "mes_atendimento" if "mes_atendimento" in dfF.columns else
            "data"
        )

        fig = tau_trimestral_acumulado_por_dia(
            df=dfF,                    # use dfF (filtrado e com _dt)
            date_col=date_col_for_tau,
            id_col="id_pessoa",
            tipo_col="tipo",           # ajuste se o seu nome for outro
            inter_tipos={"mensagem","mensagens","ligacao","ligacoes",
                            "atendimento","consulta","whatsapp","msg","chamada"},
                title="TAU Realizado Acumulado por date",
                )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Como as pessoas se engajam?", expanded=True):
        escala = st.radio(
            "Escala",
            options=["Absoluto (pessoas)", "Relativo (% por trimestre)"],
            horizontal=True,
            index=0
        )
        relative_mode = (escala == "Relativo (% por trimestre)")

        fig = heatmap_combos_por_trimestre_colscale(
            df=dfF,
            date_col="_dt",
            id_col="id_pessoa",
            tipo_col="tipo",
            relative=relative_mode,          # <<< passar a variável, não fixar False
            nenhum_last=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Como está a meta do trimestre?", expanded=True):
        # 1) escolha do trimestre
        quarters = sorted(dfF["_yq_lbl"].dropna().unique().tolist())
        if not quarters:
            st.info("Sem quarters após filtros.")
            st.stop()

        q_sel = st.selectbox("Trimestre", quarters, index=len(quarters)-1)

        # 2) meta do trimestre (0–100%)
        meta_perc = st.slider("Meta do trimestre (TAU %)", 0, 100, value=80, step=1)
        meta_tau = meta_perc / 100.0

        # 3) feriados (opcional)
        feriados_txt = st.text_area(
            "Feriados (opcional, 1 por linha no formato AAAA-MM-DD).",
            value="",
            help="Serão excluídos dos dias úteis."
        )
        feriados = [s.strip() for s in feriados_txt.splitlines() if s.strip()]

        # --- NORMALIZAÇÃO DE RÓTULO: '2025Q03' -> '2025Q3' ---
        # (aceita tanto 2025Q01..Q04 quanto 2025Q1..Q4)
        q_sel_norm = re.sub(r"Q0([1-4])$", r"Q\1", q_sel)

        fig, kpis = meta_trimestral_acumulada(
            df=dfF,
            date_col="_dt",
            id_col="id_pessoa",
            tipo_col="tipo",
            inter_tipos={"mensagem","ligacao","atendimento","whatsapp","msg","chamada","consulta"},
            quarter_label=q_sel,
            meta_tau=meta_tau,
            feriados=feriados,
            title="04 - TAU Realizado Acumulado e 09 - Meta Rateada por Dia Acumulada por date",
            return_kpis=True
        )

        # KPIs (em 2 linhas)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ativos no trimestre", f"{kpis['ativos']:,}".replace(",", "."))
        c2.metric("Engajados até a última data", f"{kpis['engajados_ate_ultima_data']:,}".replace(",", "."),
                help=f"Última data com dados: {kpis['ultima_data']}")
        c3.metric("TAU até a última data", f"{kpis['tau_ate_ultima_data']:.1%}")
        c4.metric("Alvo definido (TAU)", f"{kpis['alvo_definido_tau']:.0%}")

        c5, c6, c7, _ = st.columns(4)
        c5.metric("Faltam para bater a meta", f"{kpis['faltam_abs']:,}".replace(",", "."))
        c6.metric("Dias úteis restantes", f"{kpis['dias_uteis_restantes']}")
        c7.metric("Necessário por dia útil", f"{kpis['faltam_por_dia']:,}".replace(",", "."))

        st.plotly_chart(fig, use_container_width=True, theme=None)