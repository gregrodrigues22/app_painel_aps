# --------------------------------------------------------------- 
# Importação
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
from src.plots import (
    bar_yoy_trend,
    fig_ativos_por_mes,
    tau_trimestral_acumulado_por_dia,
    heatmap_combos_por_trimestre_colscale,
)
from src.plots import (
    meta_trimestral_acumulada,
    tau_distribuicao_mensal_por_trimestre,
    tau_distribuicao_mensal_por_trimestre_barras,
    histograma_boxplot_idade_plotly,
    pie_standard
)

# ---------------------------------------------------------------
# Configaração geral da página
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

# ---------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Ana Health — Relatório Executivo</h1>
        <p style='color: white;'>Explore os painéis para tomada de decisão baseada em dados.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Navegação entre páginas
# ---------------------------------------------------------------
def safe_page_link(path: str, label: str, icon: str | None = None):
    try:
        if (APP_DIR.parent / path).exists() or (APP_DIR / path).exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(
                label,
                icon=icon,
                disabled=True,
                help="Página não disponível neste app.",
            )
    except Exception:
        st.button(
            label,
            icon=icon,
            disabled=True,
            help="Navegação multipage indisponível aqui.",
        )


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.header("Menu")
    safe_page_link("app.py", label="Voltar ao início", icon="↩️")
    st.caption("Atalhos")
    safe_page_link("pages/relatorio_gestor.py", label="Relatório Tático", icon="📈")
    safe_page_link("pages/relatorio_operacao.py", label="Relatório Operação", icon="🏨")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Conecte-se")
    st.markdown(
        """
- 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
- ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
- 📸 [Instagram](https://www.instagram.com/patients2python/)
- 🌐 [Site](https://patients2python.com.br/)
- 🐙 [GitHub](https://github.com/gregrodrigues22)
- 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
- 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
- 🎓 [Escola](https://app.patients2python.com.br/browse)
    """,
        unsafe_allow_html=True,
    )

# =========================
# Helpers de dados
# =========================
EXPECTED_SCHEMA = {
    # coluna           (tipo esperado, descrição/valid.)
    "id_pessoa": ("string", "Identificador (texto)"),
    "sexo": ("int", "0/1 (ou 1/2) — será normalizado para 0/1"),
    "faixa_etaria": ("int", "inteiro (código de faixa)"),
    "higido": ("int01", "0/1"),
    "multicomorbido": ("int01", "0/1"),
    "diabetes": ("int01", "0/1"),
    "dislipidemia": ("int01", "0/1"),
    "hipertensao": ("int01", "0/1"),
    "obesidade": ("int01", "0/1"),
    "saude_mental": ("int01", "0/1"),
    "data": ("date", "data no formato AAAA-MM-DD"),
    "mes_atendimento": ("date", "primeiro dia do mês AAAA-MM-DD"),
    "tipo": ("string", "texto (ex.: atendimento)"),
}

COND_COLS = [
    "higido",
    "multicomorbido",
    "diabetes",
    "dislipidemia",
    "hipertensao",
    "obesidade",
    "saude_mental",
]


@st.cache_data(show_spinner=False)
def _read_csv_smart(
    file, force_sep: str | None = None, dtype_map: dict | None = None
) -> pd.DataFrame:
    dtype_map = dtype_map or {}
    if force_sep:
        return pd.read_csv(file, sep=force_sep, dtype=dtype_map)
    head = file.getvalue().splitlines()[0].decode("utf-8", errors="ignore")
    guess = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.BytesIO(file.getvalue()), sep=guess, dtype=dtype_map)


def schema_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coluna": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_null": [df[c].isna().sum() for c in df.columns],
            "exemplo": [
                df[c].dropna().iloc[0] if df[c].notna().any() else None
                for c in df.columns
            ],
        }
    )


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
        s = s.replace({2: 0})
        df["sexo"] = s.fillna(0).astype(int)

    # coerções por schema
    for col, (expected, _) in EXPECTED_SCHEMA.items():
        status = "ok"
        details = ""
        present = col in df.columns

        if not present:
            status = "faltando"
            report_rows.append(
                (col, expected, "-", status, "Coluna ausente no CSV")
            )
            continue

        try:
            if expected == "string":
                df[col] = df[col].astype("string")
            elif expected == "int":
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .astype("Int64")
                    .fillna(0)
                    .astype(int)
                )
            elif expected == "int01":
                s = pd.to_numeric(df[col], errors="coerce").fillna(0)
                # qualq. valor >0 vira 1
                df[col] = (s > 0).astype(int)
            elif expected == "date":
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce", utc=True)
                    .dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                )
            else:
                # fallback
                df[col] = df[col]
        except Exception as e:
            status = "erro"
            details = f"Falha ao converter para {expected}: {e}"

        # checagens específicas
        if expected == "int01" and present:
            invalid = (~df[col].isin([0, 1])).sum()
            if invalid > 0:
                status = "alerta"
                details = f"{invalid} valores não binários (ajustados para 0/1)."

        if expected == "date" and present:
            n_na = df[col].isna().sum()
            if n_na > 0:
                status = "alerta" if status == "ok" else status
                details = (details + " " if details else "") + (
                    f"{n_na} datas inválidas (NaT)."
                )

        report_rows.append(
            (col, expected, str(df[col].dtype), status, details.strip())
        )

    # colunas extras
    extras = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    for c in extras:
        report_rows.append(
            (c, "-", str(df[c].dtype), "extra", "Coluna não esperada (mantida).")
        )

    report = pd.DataFrame(
        report_rows, columns=["coluna", "esperado", "dtype_final", "status", "detalhes"]
    )
    return df, report

def remember_state(key: str, value):
    """
    Guarda um valor em st.session_state e retorna True se mudou
    desde a última execução.
    """
    if key not in st.session_state:
        st.session_state[key] = value
        return True
    if st.session_state[key] != value:
        st.session_state[key] = value
        return True
    return False

# ---------------------------------------------------------------
# Funções acessórias
# ---------------------------------------------------------------

def compute_heatmaps_combos_both(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    tipo_col: str,
    nenhum_last: bool,
):
    """
    Calcula uma vez o heatmap em valor absoluto e relativo
    e devolve as duas figuras.
    O controle de 'não recalcular à toa' é feito via st.session_state,
    não via st.cache_data (para não hashear DataFrame grande).
    """
    fig_abs = heatmap_combos_por_trimestre_colscale(
        df=df,
        date_col=date_col,
        id_col=id_col,
        tipo_col=tipo_col,
        relative=False,
        nenhum_last=nenhum_last,
    )

    fig_rel = heatmap_combos_por_trimestre_colscale(
        df=df,
        date_col=date_col,
        id_col=id_col,
        tipo_col=tipo_col,
        relative=True,
        nenhum_last=nenhum_last,
    )

    return fig_abs, fig_rel

@st.cache_data(show_spinner=False)
def prepare_tau_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara colunas derivadas usadas em toda a aba de engajamento.
    Cacheado para evitar refazer esse processamento a cada interação.
    """
    df = df.copy()

    # escolhe a coluna de data base
    if "mes_atendimento" in df.columns:
        base_date_col = "mes_atendimento"
    elif "data" in df.columns:
        base_date_col = "data"
    else:
        raise ValueError("Não encontrei coluna de data (`mes_atendimento` ou `data`).")

    df["_dt"] = pd.to_datetime(df[base_date_col], errors="coerce")
    df["_year"] = df["_dt"].dt.year
    df["_month_period"] = df["_dt"].dt.to_period("M")
    df["_quarter"] = df["_dt"].dt.quarter
    df["_yq_lbl"] = (
        df["_dt"].dt.year.astype(str)
        + "Q"
        + df["_dt"].dt.quarter.astype(str).str.zfill(2)
    )

    import unicodedata

    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(
            ch
            for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )
        return s

    if "tipo" in df.columns:
        df["_tipo_norm"] = df["tipo"].map(_norm)
    else:
        df["_tipo_norm"] = pd.Series("", index=df.index)

    return df

def calcular_desassistidos_6m_por_trimestre(dfF, inter_tipos):
    """
    Calcula, para cada trimestre (YQ):
        - Pessoas ativas no trimestre
        - Pessoas com interações nos últimos 6 meses
        - Pessoas desassistidas (nenhuma interação nos últimos 6 meses)

    Retorna um pandas.Series onde o índice é YYYYQX e o valor = nº de desassistidos.
    """

    # Apenas interações válidas
    df_inter = dfF[dfF["_tipo_norm"].isin(inter_tipos)].copy()

    # Pessoas ativas por trimestre
    ativos_q = (
        dfF.groupby("_yq_lbl")["id_pessoa"]
        .nunique()
        .rename("ativos_q")
    )

    # Fim de cada trimestre
    quarter_end = dfF.groupby("_yq_lbl")["_dt"].max()

    desassistidos_dict = {}

    for q in ativos_q.index:
        end = quarter_end.get(q, pd.NaT)
        if pd.isna(end):
            continue

        # janela de 6 meses
        start = end - pd.DateOffset(months=6)

        # pessoas ativas no trimestre q
        ids_ativos_q = (
            dfF.loc[dfF["_yq_lbl"] == q, "id_pessoa"]
            .dropna()
            .unique()
        )

        # pessoas com interações dentro da janela 6m
        ids_eng_6m = (
            df_inter.loc[
                (df_inter["_dt"] > start) & (df_inter["_dt"] <= end),
                "id_pessoa",
            ]
            .dropna()
            .unique()
        )

        # desassistidos = ativos no trimestre sem interações na janela
        desassistidos = len(set(ids_ativos_q) - set(ids_eng_6m))
        desassistidos_dict[q] = desassistidos

    return pd.Series(desassistidos_dict).sort_index()

def kpis_desassistidos(series_des):
    if series_des.empty:
        return 0, 0, 0, 0

    # Normalizar índice YYYYQX → PeriodIndex
    idx_norm = (
        series_des.index.to_series()
        .str.replace(r"Q0([1-4])$", r"Q\1", regex=True)
    )
    ordered = series_des.copy()
    ordered.index = pd.PeriodIndex(idx_norm, freq="Q")
    ordered = ordered.sort_index()

    max_v = int(ordered.max())
    min_v = int(ordered.min())
    mean_v = float(ordered.mean())
    last_v = int(ordered.iloc[-1])

    return max_v, min_v, mean_v, last_v

def hist_interacoes_por_pessoa(
    df: pd.DataFrame,
    col_numerica: str = "n_interacoes",
    nbins: int = 20,
    titulo: str = "Distribuição de interações por pessoa",
):
    """
    Wrapper que reaproveita histograma+boxplot de idade
    para mostrar distribuição de interações por pessoa.
    """
    # usa a função genérica que você já tem
    fig = histograma_boxplot_idade_plotly(
        df=df,
        col_numerica=col_numerica,
        nbins=nbins,
        largura_px=1100,
        altura_px=650,
        titulo=titulo,
    )

    # Ajusta rótulos de eixos
    fig.update_xaxes(title_text="Nº de interações por pessoa", row=1, col=1)
    fig.update_yaxes(title_text="Número de pessoas",            row=1, col=1)
    fig.update_xaxes(title_text="Nº de interações por pessoa", row=2, col=1)

    # Ajusta nomes no hover/legenda para não ficar “idade/alunos”
    for trace in fig.data:
        if getattr(trace, "name", None) == "Idade":
            trace.name = "Interações por pessoa"
        if getattr(trace, "hovertemplate", None):
            trace.hovertemplate = trace.hovertemplate.replace("Alunos", "Pessoas")

    return fig

def preparar_pizza_interacoes_zero_vs_maior_zero(
    dfF: pd.DataFrame,
    *,
    tipos: set,
    somente_ativos_hoje: bool = False,
) -> pd.DataFrame:
    """
    Retorna um DF com 2 linhas:
        grupo  | qtd
        ----------------------------
        'Com interações'  | N1
        'Sem interações'  | N0

    considerando:
      - dfF: base de interações já filtrada pelos filtros globais;
      - tipos: conjunto de tipos de interação para considerar;
      - somente_ativos_hoje: se True, universo = pessoas ativas no último _month_period.
    """

    if dfF.empty or "id_pessoa" not in dfF.columns:
        return pd.DataFrame({"grupo": [], "qtd": []})

    # 1) universo de pessoas
    if somente_ativos_hoje:
        ultimo_mes = dfF["_month_period"].max()
        ids_base = (
            dfF.loc[dfF["_month_period"] == ultimo_mes, "id_pessoa"]
            .dropna()
            .unique()
        )
    else:
        ids_base = dfF["id_pessoa"].dropna().unique()

    base_pessoas = pd.DataFrame({"id_pessoa": ids_base})

    # 2) filtra apenas tipos de interação desejados
    df_int = dfF[dfF["_tipo_norm"].isin(tipos)].copy()

    # 3) conta interações por pessoa (pode ser 0 se não tiver linha)
    contagens = (
        df_int.groupby("id_pessoa")
        .size()
        .rename("n_interacoes")
    )

    base_pessoas = base_pessoas.merge(
        contagens, on="id_pessoa", how="left"
    )
    base_pessoas["n_interacoes"] = base_pessoas["n_interacoes"].fillna(0)

    # 4) classifica em 0 vs >0
    base_pessoas["grupo"] = np.where(
        base_pessoas["n_interacoes"] > 0,
        "Com interações",
        "Sem interações",
    )

    df_pizza = (
        base_pessoas.groupby("grupo", as_index=False)["id_pessoa"]
        .nunique()
        .rename(columns={"id_pessoa": "qtd"})
    )

    return df_pizza

# =========================
# Abas
# =========================
tab1, tab2, tab3 = st.tabs(
    ["📥 Importar & Validar", "📊 Engajamento", "😭 Desassistência"]
)

# ---------------------------------------------------------------
# Aba 1 
# ---------------------------------------------------------------
with tab1:
    st.subheader("📤 Carregue seu CSV")
    st.write("Formato esperado e tipos por coluna:")

    sch = pd.DataFrame(
        [
            {"coluna": c, "tipo_esperado": t[0], "descrição": t[1]}
            for c, t in EXPECTED_SCHEMA.items()
        ]
    )
    st.dataframe(sch, use_container_width=True, hide_index=True)

    uploaded = st.file_uploader(
        "Arraste e solte o arquivo aqui, ou clique para selecionar",
        type=["csv", "txt"],
        accept_multiple_files=False,
        help="Tamanho máximo por arquivo: ~200MB",
    )

    sep_override = st.radio(
        "Separador",
        options=["Detectar automaticamente", "Vírgula (,)", "Ponto e vírgula (;)"],
        horizontal=True,
    )
    force_sep = None
    if sep_override == "Vírgula (,)":
        force_sep = ","
    elif sep_override == "Ponto e vírgula (;)":
        force_sep = ";"

    if uploaded is not None:
        with st.spinner("⏳ Carregando e validando..."):
            try:
                df_raw = _read_csv_smart(
                    uploaded, force_sep=force_sep, dtype_map={"id_pessoa": "string"}
                )
                df, rep = validate_and_clean(df_raw)

                st.success(
                    f"✅ Arquivo carregado: **{uploaded.name}** — {df.shape[0]:,} linhas × {df.shape[1]} colunas".replace(
                        ",", "."
                    )
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Schema detectado**")
                    st.dataframe(
                        schema_df(df), use_container_width=True, hide_index=True
                    )
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
# ABA 2
# =========================
with tab2:
    st.subheader("📊 Engajamento")

    if "df_tau" not in st.session_state:
        st.warning(
            "Nenhum dado validado disponível. Vá à aba **Importar & Validar** e carregue o CSV."
        )
        st.stop()

    # ---------- PREP BÁSICO (cacheado) ----------
    df_base = st.session_state["df_tau"]
    try:
        df = prepare_tau_base(df_base)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    INTER_TIPOS = {"mensagem", "ligacao", "atendimento"}
    INTER_TIPOS_MS = {"mensagem"}
    INTER_TIPOS_LG = {"ligacao"}
    INTER_TIPOS_AT = {"atendimento"}

    COND_COLS = [
        c
        for c in [
            "higido",
            "multicomorbido",
            "diabetes",
            "dislipidemia",
            "hipertensao",
            "obesidade",
            "saude_mental",
        ]
        if c in df.columns
    ]

    # ------------------------------
    # Sessão de Filtros
    # ------------------------------
    st.info(
        "🧪 Sessão de Filtros: Use os controles abaixo para refinar os resultados."
    )

    with st.container(border=True):
        # Linha 1 — Sexo | Faixa Etária
        c1, c2 = st.columns(2)
        sexo_opts = (
            ["(Todos)"]
            + sorted(map(str, df["sexo"].dropna().unique()))
            if "sexo" in df.columns
            else ["(N/A)"]
        )
        sexo_sel = c1.selectbox("Sexo", sexo_opts, index=0)

        fe_opts = (
            ["(Todos)"]
            + sorted(map(str, df["faixa_etaria"].dropna().unique()))
            if "faixa_etaria" in df.columns
            else ["(N/A)"]
        )
        fe_sel = c2.selectbox("Faixa Etária", fe_opts, index=0)

        # Linha 2 — checkboxes das condições
        st.markdown("**Condições**")
        cA, cB, cC, cD, cE, cF, cG = st.columns(7)
        chk_alguma = cA.checkbox(
            "Alguma condição",
            value=False,
            help="Filtra quem tem pelo menos 1 condição verdadeira",
        )
        chk_multi = (
            cB.checkbox("Multicomorbido", value=False)
            if "multicomorbido" in df.columns
            else (False)
        )
        chk_diab = (
            cC.checkbox("Diabetes", value=False)
            if "diabetes" in df.columns
            else (False)
        )
        chk_disl = (
            cD.checkbox("Dislidepidemia", value=False)
            if "dislipidemia" in df.columns
            else (False)
        )
        chk_hip = (
            cE.checkbox("Hipertensão", value=False)
            if "hipertensao" in df.columns
            else (False)
        )
        chk_obes = (
            cF.checkbox("Obesidade", value=False)
            if "obesidade" in df.columns
            else (False)
        )
        chk_sm = (
            cG.checkbox("Saúde Mental", value=False)
            if "saude_mental" in df.columns
            else (False)
        )

        # Linha 3 — Período (ano)
        cY = st.columns([1])[0]
        minY, maxY = int(df["_year"].min()), int(df["_year"].max())
        yr1, yr2 = cY.slider(
            "Período (ano)", min_value=minY, max_value=maxY, value=(minY, maxY)
        )

        # Linha 4 — Trimestre (YYYYQQ)
        colQ, _ = st.columns([1, 1])
        yq_all = sorted(df["_yq_lbl"].dropna().unique().tolist())
        if not yq_all:
            yq_all = []
        yq_sel = colQ.multiselect(
            "Trimestre (YYYYQQ)",
            yq_all,
            default=yq_all,
            help="Filtra por trimestres. Combine com o intervalo de anos ao lado.",
        )

    # ------ aplica filtros de seleção ------
    mask = (df["_year"].between(yr1, yr2)) & (df["_yq_lbl"].isin(yq_sel))
    if sexo_sel != "(Todos)" and "sexo" in df.columns:
        mask &= df["sexo"].astype(str) == sexo_sel
    if fe_sel != "(Todos)" and "faixa_etaria" in df.columns:
        mask &= df["faixa_etaria"].astype(str) == fe_sel

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

        pessoas_dist = (
            dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        )

        if not dfF.empty:
            last_month = dfF["_month_period"].max()
            pessoas_ativas = dfF.loc[
                dfF["_month_period"] == last_month, "id_pessoa"
            ].nunique()
        else:
            pessoas_ativas = 0

        meses_dist = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        media_pessoa_mes = float(pessoas_dist / max(1, meses_dist))
        quarters_dist = int(dfF["_yq_lbl"].nunique()) if not dfF.empty else 0
        media_pessoa_quarters = float(pessoas_dist / max(1, quarters_dist))

        def fmt_br(x, nd=2):
            s = f"{x:,.{nd}f}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")

        c1.metric("Pessoas distintas", f"{pessoas_dist:,}".replace(",", "."))
        c2.metric(
            "Pessoas ativas (último período)",
            f"{pessoas_ativas:,}".replace(",", "."),
        )
        c3.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_mes))
        c4.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_quarters))

        st.markdown("---")

        # ========= ENGAJAMENTO TRIMESTRAL (4 KPIs) =========
        # TAU por trimestre = pessoas com pelo menos 1 interação / pessoas ativas no trimestre
        if "id_pessoa" in dfF.columns and "_yq_lbl" in dfF.columns:
            # 1) pessoas ativas por trimestre
            base_q = (
                dfF.groupby("_yq_lbl")["id_pessoa"]
                .nunique()
                .rename("ativos_q")
                .to_frame()
            )

            # 2) pessoas engajadas por trimestre (pelo menos 1 interação)
            inter_mask = dfF["_tipo_norm"].isin(INTER_TIPOS)
            engajados_q = (
                dfF[inter_mask]
                .groupby("_yq_lbl")["id_pessoa"]
                .nunique()
                .rename("engajados_q")
            )

            base_q = base_q.join(engajados_q, how="left").fillna(0)

            base_q["tau_q"] = base_q["engajados_q"] / base_q["ativos_q"].replace(0, np.nan)
            serie_tau = base_q["tau_q"].dropna()
        else:
            serie_tau = pd.Series(dtype=float)

        if not serie_tau.empty:
            tau_max = float(serie_tau.max())
            tau_min = float(serie_tau.min())
            tau_mean = float(serie_tau.mean())

            # ordenar trimestres cronologicamente para pegar o "último engajamento"
            # seus rótulos são do tipo "2024Q01" → normalizamos para "2024Q1"
            idx_norm = (
                serie_tau.index.to_series()
                .str.replace(r"Q0([1-4])$", r"Q\1", regex=True)
            )
            serie_tau_ord = serie_tau.copy()
            serie_tau_ord.index = pd.PeriodIndex(idx_norm, freq="Q")
            tau_last = float(serie_tau_ord.sort_index().iloc[-1])
        else:
            tau_max = tau_min = tau_mean = tau_last = 0.0

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Maior engajamento trimestral (TAU)", f"{tau_max:.1%}")
        e2.metric("Menor engajamento trimestral (TAU)", f"{tau_min:.1%}")
        e3.metric("Engajamento trimestral médio (TAU)", f"{tau_mean:.1%}")
        e4.metric("Engajamento no último trimestre (TAU)", f"{tau_last:.1%}")

        # ========= TOPO (4 KPIs) =========
        #c5, c6, c7, c8 = st.columns(4)

        #pessoas_dist = (
        #    dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        #)

        #if not dfF.empty:
        #    last_month = dfF["_month_period"].max()
        #    pessoas_ativas = dfF.loc[
        #        dfF["_month_period"] == last_month, "id_pessoa"
        #    ].nunique()
        #else:
        #    pessoas_ativas = 0

        #df_inter = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS)]
        #num_inter = int(len(df_inter))
        #media_inter = float(num_inter / max(1, pessoas_dist))
        #df_mensagens = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_MS)]
        #num_mensagens = int(len(df_mensagens))
        #media_mensagens = float(num_mensagens / max(1, pessoas_dist))

        #c5.metric(
        #    "Número de interações Totais", f"{num_inter:,}".replace(",", ".")
        #)
        #c6.metric(
        #    "Média de interações Totais por pessoa",
        #    f"{media_inter:,.2f}".replace(",", "."),
        #)
        #c7.metric(
        #    "Número de interações Mensagens",
        #    f"{num_mensagens:,}".replace(",", "."),
        #)
        #c8.metric(
        #    "Média de interações Mensagens por pessoa",
        #    f"{media_mensagens:,.2f}".replace(",", "."),
        #)

        # ========= TOPO (4 KPIs) =========
        #c9, c10, c11, c12 = st.columns(4)

        #pessoas_dist = (
        #    dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        #)

        #if not dfF.empty:
        #    last_month = dfF["_month_period"].max()
        #    pessoas_ativas = dfF.loc[
        #        dfF["_month_period"] == last_month, "id_pessoa"
        #    ].nunique()
        #else:
        #    pessoas_ativas = 0

        #df_agendamentos = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_AT)]
        #num_agendamentos = int(len(df_agendamentos))
        #media_agendamentos = float(num_agendamentos / max(1, pessoas_dist))
        #df_ligacoes = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_LG)]
        #num_ligacoes = int(len(df_ligacoes))
        #media_ligacoes = float(num_ligacoes / max(1, pessoas_dist))

        #c9.metric(
        #    "Número de interações Agendamento",
        #    f"{num_agendamentos:,}".replace(",", "."),
        #)
        #c10.metric(
        #    "Média de interações Agendamento por pessoa",
        #    f"{media_agendamentos:,.2f}".replace(",", "."),
        #)
        #c11.metric(
        #    "Número de interações Ligações",
        #    f"{num_ligacoes:,}".replace(",", "."),
        #)
        #c12.metric(
        #    "Média de interações Ligações por pessoa",
        #    f"{media_ligacoes:,.2f}".replace(",", "."),
        #)

        #st.markdown("---")

        # ========= CONDIÇÕES (4 + 3) =========
        #if "id_pessoa" in dfF.columns and COND_COLS:
        #    base_pessoa = dfF.groupby("id_pessoa")[COND_COLS].max(numeric_only=True)
        #    n_alguma = int(base_pessoa.any(axis=1).sum())
        #    n_multi = int(
        #        base_pessoa.get(
        #            "multicomorbido",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_diab = int(
        #        base_pessoa.get(
        #            "diabetes", pd.Series(False, index=base_pessoa.index)
        #        ).sum()
        #    )
        #    n_disl = int(
        #        base_pessoa.get(
        #            "dislipidemia",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_hip = int(
        #        base_pessoa.get(
        #            "hipertensao",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_obes = int(
        #        base_pessoa.get(
        #            "obesidade", pd.Series(False, index=base_pessoa.index)
        #        ).sum()
        #    )
        #    n_sm = int(
        #        base_pessoa.get(
        #            "saude_mental",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #else:
        #    n_alguma = n_multi = n_diab = n_disl = n_hip = n_obes = n_sm = 0

        # Linha 1 (4 KPIs de condição)
        #k1, k2, k3, k4 = st.columns(4)
        #k1.metric("Com alguma condição", f"{n_alguma:,}".replace(",", "."))
        #k2.metric("Multicomorbidos", f"{n_multi:,}".replace(",", "."))
        #k3.metric("Diabéticos", f"{n_diab:,}".replace(",", "."))
        #k4.metric("Dislidepidêmicos", f"{n_disl:,}".replace(",", "."))

        # Linha 2 (3 KPIs de condição)
        #k5, k6, k7, k8 = st.columns(4)
        #k5.metric("Hipertensos", f"{n_hip:,}".replace(",", "."))
        #k6.metric("Obesos", f"{n_obes:,}".replace(",", "."))
        #k7.metric("Pessoas saúde mental", f"{n_sm:,}".replace(",", "."))

        #st.markdown("---")

        # ========= TEMPO (3 KPIs) =========
        #t1, t2, t3, t4 = st.columns(4)
        #meses_dist = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        #anos_dist = int(dfF["_year"].nunique()) if not dfF.empty else 0
        #quarters_dist = int(dfF["_yq_lbl"].nunique()) if not dfF.empty else 0

        #t1.metric("Meses distintos", f"{meses_dist:,}".replace(",", "."))
        #t2.metric("Anos distintos", f"{anos_dist:,}".replace(",", "."))
        #t3.metric("Quarters distintos", f"{quarters_dist:,}".replace(",", "."))

        #st.markdown("---")

    # ------------------------------
    # Sessão de Gráficos
    # ------------------------------
    st.info(
        "📈 Sessão de Gráficos: visuais para responder às perguntas segundo os filtros aplicados."
    )

    with st.expander("Qual número de pessoas ativas por mês?", expanded=False):
        fig = fig_ativos_por_mes(
            dfF,  # seu dataframe filtrado
            date_col="_dt",  # ou "mes_atendimento" se você quiser usar essa coluna
            id_col="id_pessoa",
            trend_alpha=0.10,  # mais baixo = mais suave
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("Qual é o nível engajamento por trimestre?", expanded=False):
        fig = tau_trimestral_acumulado_por_dia(
            df=dfF,
            date_col="data",  # a função faz fallback se precisar
            id_col="id_pessoa",
            tipo_col="tipo",
            inter_tipos={
                "mensagem",
                "mensagens",
                "ligacao",
                "ligacoes",
                "atendimento",
                "consulta",
                "whatsapp",
                "msg",
                "chamada",
            },
            title="TAU Realizado Acumulado por date",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Quando o engajamento acontece dentro do trimestre?", expanded=False):
        fig_barras = tau_distribuicao_mensal_por_trimestre_barras(
            df=dfF,
            date_col="data",
            id_col="id_pessoa",
            tipo_col="tipo",
            inter_tipos={
                "mensagem",
                "mensagens",
                "ligacao",
                "ligacoes",
                "atendimento",
                "consulta",
                "whatsapp",
                "msg",
                "chamada",
            },
            title="Distribuição do engajamento por mês do trimestre",
        )
        st.plotly_chart(fig_barras, use_container_width=True)

    with st.expander("Como as pessoas se engajam?", expanded=False):
        escala = st.radio(
            "Escala",
            options=["Absoluto (pessoas)", "Relativo (% por trimestre)"],
            horizontal=True,
            index=0,
            key="tau_heatmap_escala",
        )

        # ------------------------------
        # Detectar se filtros MUDARAM
        # (quando só muda a escala, isso aqui continua igual)
        # ------------------------------
        heatmap_filters = {
            "yr1": yr1,
            "yr2": yr2,
            "yq_sel": tuple(yq_sel),
            "sexo_sel": sexo_sel,
            "fe_sel": fe_sel,
            "chk_alguma": chk_alguma,
            "chk_multi": chk_multi,
            "chk_diab": chk_diab,
            "chk_disl": chk_disl,
            "chk_hip": chk_hip,
            "chk_obes": chk_obes,
            "chk_sm": chk_sm,
        }

        filters_changed_for_heatmap = remember_state(
            "__tau_heatmap_filters", heatmap_filters
        )

        # ------------------------------
        # Só recalcula heatmap se filtros mudaram
        # (não por causa da escala)
        # ------------------------------
        if "tau_heatmap_figs" not in st.session_state or filters_changed_for_heatmap:
            with st.spinner("Calculando heatmap de engajamento..."):
                fig_abs, fig_rel = compute_heatmaps_combos_both(
                    df=dfF,
                    date_col="data",   # ou "_dt" se preferir
                    id_col="id_pessoa",
                    tipo_col="tipo",
                    nenhum_last=False,
                )
            st.session_state["tau_heatmap_figs"] = (fig_abs, fig_rel)
        else:
            fig_abs, fig_rel = st.session_state["tau_heatmap_figs"]

        # Agora só escolhe qual exibir, sem recalcular nada
        if escala == "Relativo (% por trimestre)":
            fig = fig_rel
        else:
            fig = fig_abs

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Como está a meta do trimestre?", expanded=False):
        # 1) escolha do trimestre
        quarters = sorted(dfF["_yq_lbl"].dropna().unique().tolist())
        if not quarters:
            st.info("Sem quarters após filtros.")
            st.stop()

        q_sel = st.selectbox("Trimestre", quarters, index=len(quarters) - 1)

        # 2) meta do trimestre (0–100%)
        meta_perc = st.slider("Meta do trimestre (TAU %)", 0, 100, value=80, step=1)
        meta_tau = meta_perc / 100.0

        # 3) feriados (opcional)
        feriados_txt = st.text_area(
            "Feriados (opcional, 1 por linha no formato AAAA-MM-DD).",
            value="",
            help="Serão excluídos dos dias úteis.",
        )
        feriados = [s.strip() for s in feriados_txt.splitlines() if s.strip()]

        # normalização do rótulo
        q_sel_norm = re.sub(r"Q0([1-4])$", r"Q\1", q_sel)

        fig_meta, kpis = meta_trimestral_acumulada(
            df=dfF,
            date_col="data",
            id_col="id_pessoa",
            tipo_col="tipo",
            inter_tipos={
                "mensagem",
                "ligacao",
                "atendimento",
                "whatsapp",
                "msg",
                "chamada",
                "consulta",
            },
            quarter_label=q_sel_norm,
            meta_tau=meta_tau,
            feriados=feriados,
            title="04 - TAU Realizado Acumulado e 09 - Meta Rateada por Dia Acumulada por date",
            return_kpis=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ativos no trimestre", f"{kpis['ativos']:,}".replace(",", "."))
        c2.metric(
            "Engajados até a última data",
            f"{kpis['engajados_ate_ultima_data']:,}".replace(",", "."),
            help=f"Última data com dados: {kpis['ultima_data']}",
        )
        c3.metric("TAU até a última data", f"{kpis['tau_ate_ultima_data']:.1%}")
        c4.metric("Alvo definido (TAU)", f"{kpis['alvo_definido_tau']:.0%}")

        c5, c6, c7, _ = st.columns(4)
        c5.metric(
            "Faltam para bater a meta",
            f"{kpis['faltam_abs']:,}".replace(",", "."),
        )
        c6.metric("Dias úteis restantes", f"{kpis['dias_uteis_restantes']}")
        c7.metric(
            "Necessário por dia útil",
            f"{kpis['faltam_por_dia']:,}".replace(",", "."),
        )

        st.plotly_chart(fig_meta, use_container_width=True, theme=None)

# ---------------------------------------------------------------
# Aba 3
# ---------------------------------------------------------------

with tab3:
    st.subheader("😭 Desassistência")

    if "df_tau" not in st.session_state:
        st.warning(
            "Nenhum dado validado disponível. Vá à aba **Importar & Validar** e carregue o CSV."
        )
        st.stop()
    if "df_tau" not in st.session_state:
        st.warning(
            "Nenhum dado validado disponível. Vá à aba **Importar & Validar** e carregue o CSV."
        )
        st.stop()

    # ---------- PREP BÁSICO (cacheado) ----------
    df_base = st.session_state["df_tau"]
    try:
        df = prepare_tau_base(df_base)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    INTER_TIPOS = {"mensagem", "ligacao", "atendimento"}
    INTER_TIPOS_MS = {"mensagem"}
    INTER_TIPOS_LG = {"ligacao"}
    INTER_TIPOS_AT = {"atendimento"}

    COND_COLS = [
        c
        for c in [
            "higido",
            "multicomorbido",
            "diabetes",
            "dislipidemia",
            "hipertensao",
            "obesidade",
            "saude_mental",
        ]
        if c in df.columns
    ]

    # ------------------------------
    # Sessão de Filtros
    # ------------------------------
    st.info(
        "🧪 Sessão de Filtros: Use os controles abaixo para refinar os resultados."
    )

    with st.container(border=True):
        # Linha 1 — Sexo | Faixa Etária
        c1, c2 = st.columns(2)
        sexo_opts = (
            ["(Todos)"]
            + sorted(map(str, df["sexo"].dropna().unique()))
            if "sexo" in df.columns
            else ["(N/A)"]
        )
        sexo_sel = c1.selectbox("Sexo", sexo_opts, index=0, key="desassist_sexo",)

        fe_opts = (
            ["(Todos)"]
            + sorted(map(str, df["faixa_etaria"].dropna().unique()))
            if "faixa_etaria" in df.columns
            else ["(N/A)"]
        )
        fe_sel = c2.selectbox("Faixa Etária", fe_opts, index=0, key="desassist_faixa_etaria",)

        # Linha 2 — checkboxes das condições
        st.markdown("**Condições**")
        cA, cB, cC, cD, cE, cF, cG = st.columns(7)
        chk_alguma = cA.checkbox(
            "Alguma condição",
            value=False,
            help="Filtra quem tem pelo menos 1 condição verdadeira",
            key="desassist_alguma_cond"
        )
        chk_multi = (
            cB.checkbox("Multicomorbido", value=False, key="desassist_multicomorbido")
            if "multicomorbido" in df.columns
            else (False)
        )
        chk_diab = (
            cC.checkbox("Diabetes", value=False, key="desassist_diabetes")
            if "diabetes" in df.columns
            else (False)
        )
        chk_disl = (
            cD.checkbox("Dislidepidemia", value=False, key="desassist_dislipidemia")
            if "dislipidemia" in df.columns
            else (False)
        )
        chk_hip = (
            cE.checkbox("Hipertensão", value=False, key="desassist_hipertensao")
            if "hipertensao" in df.columns
            else (False)
        )
        chk_obes = (
            cF.checkbox("Obesidade", value=False, key="desassist_obesidade")
            if "obesidade" in df.columns
            else (False)
        )
        chk_sm = (
            cG.checkbox("Saúde Mental", value=False, key="desassist_saude_mental")
            if "saude_mental" in df.columns
            else (False)
        )

        # Linha 3 — Período (ano)
        cY = st.columns([1])[0]
        minY, maxY = int(df["_year"].min()), int(df["_year"].max())
        yr1, yr2 = cY.slider(
            "Período (ano)", min_value=minY, max_value=maxY, value=(minY, maxY), key="desassist_periodo_ano"
        )

        # Linha 4 — Trimestre (YYYYQQ)
        colQ, _ = st.columns([1, 1])
        yq_all = sorted(df["_yq_lbl"].dropna().unique().tolist())
        if not yq_all:
            yq_all = []
        yq_sel = colQ.multiselect(
            "Trimestre (YYYYQQ)",
            yq_all,
            default=yq_all,
            help="Filtra por trimestres. Combine com o intervalo de anos ao lado.",
            key="desassist_trimestre"
        )

    # ------ aplica filtros de seleção ------
    mask = (df["_year"].between(yr1, yr2)) & (df["_yq_lbl"].isin(yq_sel))
    if sexo_sel != "(Todos)" and "sexo" in df.columns:
        mask &= df["sexo"].astype(str) == sexo_sel
    if fe_sel != "(Todos)" and "faixa_etaria" in df.columns:
        mask &= df["faixa_etaria"].astype(str) == fe_sel

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

        pessoas_dist = (
            dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        )

        if not dfF.empty:
            last_month = dfF["_month_period"].max()
            pessoas_ativas = dfF.loc[
                dfF["_month_period"] == last_month, "id_pessoa"
            ].nunique()
        else:
            pessoas_ativas = 0

        meses_dist = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        media_pessoa_mes = float(pessoas_dist / max(1, meses_dist))
        quarters_dist = int(dfF["_yq_lbl"].nunique()) if not dfF.empty else 0
        media_pessoa_quarters = float(pessoas_dist / max(1, quarters_dist))

        def fmt_br(x, nd=2):
            s = f"{x:,.{nd}f}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")

        c1.metric("Pessoas distintas", f"{pessoas_dist:,}".replace(",", "."))
        c2.metric(
            "Pessoas ativas (último período)",
            f"{pessoas_ativas:,}".replace(",", "."),
        )
        c3.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_mes))
        c4.metric("Pessoas distintas ativas por mês", fmt_br(media_pessoa_quarters))

        st.markdown("---")

        serie_des = calcular_desassistidos_6m_por_trimestre(
            dfF,
            inter_tipos=INTER_TIPOS
        )

        des_max, des_min, des_mean, des_last = kpis_desassistidos(serie_des)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Máximo desassistidos (6m)", f"{des_max:,}".replace(",", "."))
        d2.metric("Mínimo desassistidos (6m)", f"{des_min:,}".replace(",", "."))
        d3.metric("Média desassistidos (6m)", f"{des_mean:,.1f}".replace(",", "."))
        d4.metric("Último trimestre (desassistidos 6m)", f"{des_last:,}".replace(",", "."))

        # ========= TOPO (4 KPIs) =========
        #c5, c6, c7, c8 = st.columns(4)

        #pessoas_dist = (
        #    dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        #)

        #if not dfF.empty:
        #    last_month = dfF["_month_period"].max()
        #    pessoas_ativas = dfF.loc[
        #       dfF["_month_period"] == last_month, "id_pessoa"
        #    ].nunique()
        #else:
        #    pessoas_ativas = 0

        #df_inter = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS)]
        #num_inter = int(len(df_inter))
        #media_inter = float(num_inter / max(1, pessoas_dist))
        #df_mensagens = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_MS)]
        #num_mensagens = int(len(df_mensagens))
        #media_mensagens = float(num_mensagens / max(1, pessoas_dist))

        #c5.metric(
        #    "Número de interações Totais", f"{num_inter:,}".replace(",", ".")
        #)
        #c6.metric(
        #    "Média de interações Totais por pessoa",
        #    f"{media_inter:,.2f}".replace(",", "."),
        #)
        #c7.metric(
        #    "Número de interações Mensagens",
        #    f"{num_mensagens:,}".replace(",", "."),
        #)
        #c8.metric(
        #    "Média de interações Mensagens por pessoa",
        #    f"{media_mensagens:,.2f}".replace(",", "."),
        #)

        # ========= TOPO (4 KPIs) =========
        #c9, c10, c11, c12 = st.columns(4)

        #pessoas_dist = (
        #    dfF["id_pessoa"].nunique() if "id_pessoa" in dfF.columns else len(dfF)
        #)

        #if not dfF.empty:
        #    last_month = dfF["_month_period"].max()
        #    pessoas_ativas = dfF.loc[
        #        dfF["_month_period"] == last_month, "id_pessoa"
        #    ].nunique()
        #else:
        #    pessoas_ativas = 0

        #df_agendamentos = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_AT)]
        #num_agendamentos = int(len(df_agendamentos))
        #media_agendamentos = float(num_agendamentos / max(1, pessoas_dist))
        #df_ligacoes = dfF[dfF["_tipo_norm"].isin(INTER_TIPOS_LG)]
        #num_ligacoes = int(len(df_ligacoes))
        #media_ligacoes = float(num_ligacoes / max(1, pessoas_dist))

        #c9.metric(
        #    "Número de interações Agendamento",
        #    f"{num_agendamentos:,}".replace(",", "."),
        #)
        #c10.metric(
        #    "Média de interações Agendamento por pessoa",
        #    f"{media_agendamentos:,.2f}".replace(",", "."),
        #)
        #c11.metric(
        #    "Número de interações Ligações",
        #    f"{num_ligacoes:,}".replace(",", "."),
        #)
        #c12.metric(
        #    "Média de interações Ligações por pessoa",
        #    f"{media_ligacoes:,.2f}".replace(",", "."),
        #)

        #st.markdown("---")

        # ========= CONDIÇÕES (4 + 3) =========
        #if "id_pessoa" in dfF.columns and COND_COLS:
        #    base_pessoa = dfF.groupby("id_pessoa")[COND_COLS].max(numeric_only=True)
        #    n_alguma = int(base_pessoa.any(axis=1).sum())
        #    n_multi = int(
        #        base_pessoa.get(
        #            "multicomorbido",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_diab = int(
        #        base_pessoa.get(
        #            "diabetes", pd.Series(False, index=base_pessoa.index)
        #        ).sum()
        #    )
        #    n_disl = int(
        #        base_pessoa.get(
        #            "dislipidemia",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_hip = int(
        #        base_pessoa.get(
        #            "hipertensao",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #    n_obes = int(
        #        base_pessoa.get(
        #            "obesidade", pd.Series(False, index=base_pessoa.index)
        #        ).sum()
        #    )
        #    n_sm = int(
        #        base_pessoa.get(
        #            "saude_mental",
        #            pd.Series(False, index=base_pessoa.index),
        #        ).sum()
        #    )
        #else:
        #    n_alguma = n_multi = n_diab = n_disl = n_hip = n_obes = n_sm = 0

        # Linha 1 (4 KPIs de condição)
        #k1, k2, k3, k4 = st.columns(4)
        #k1.metric("Com alguma condição", f"{n_alguma:,}".replace(",", "."))
        #k2.metric("Multicomorbidos", f"{n_multi:,}".replace(",", "."))
        #k3.metric("Diabéticos", f"{n_diab:,}".replace(",", "."))
        #k4.metric("Dislidepidêmicos", f"{n_disl:,}".replace(",", "."))

        # Linha 2 (3 KPIs de condição)
        #k5, k6, k7, k8 = st.columns(4)
        #k5.metric("Hipertensos", f"{n_hip:,}".replace(",", "."))
        #k6.metric("Obesos", f"{n_obes:,}".replace(",", "."))
        #k7.metric("Pessoas saúde mental", f"{n_sm:,}".replace(",", "."))

        #st.markdown("---")

        # ========= TEMPO (3 KPIs) =========
        #t1, t2, t3, t4 = st.columns(4)
        #meses_dist = int(dfF["_month_period"].nunique()) if not dfF.empty else 0
        #anos_dist = int(dfF["_year"].nunique()) if not dfF.empty else 0
        #quarters_dist = int(dfF["_yq_lbl"].nunique()) if not dfF.empty else 0

        #t1.metric("Meses distintos", f"{meses_dist:,}".replace(",", "."))
        #t2.metric("Anos distintos", f"{anos_dist:,}".replace(",", "."))
        #t3.metric("Quarters distintos", f"{quarters_dist:,}".replace(",", "."))

    # ------------------------------
    # Sessão de Gráficos
    # ------------------------------
    st.info(
        "📈 Sessão de Gráficos: visuais para responder às perguntas segundo os filtros aplicados."
    )

    with st.expander("Como evolui o número de desassistidos (6 meses) por trimestre?", expanded=False):

        serie_des = calcular_desassistidos_6m_por_trimestre(
            dfF,
            inter_tipos=INTER_TIPOS,
        )

        if serie_des.empty:
            st.warning("Sem dados para desassistidos com os filtros atuais.")
        else:
            df_des = (
                serie_des
                .rename("desassistidos")
                .reset_index()
                .rename(columns={"index": "quarter"})
            )

            fig_des = bar_yoy_trend(
                df=df_des,
                x="quarter",
                y="desassistidos",
                title="Pessoas desassistidas (sem interação em 6 meses) por trimestre",
                x_is_year=False,           # aqui o eixo é categórico tipo '2023Q1'
                show_ma=True,
                ma_window=3,
                show_mean=True,
                show_trend=True,
                trend_alpha=0.15,
                y_label="Pessoas desassistidas",
                thousands=False,
                x_angle=45,
                legend_pos="top",
            )

            st.plotly_chart(fig_des, use_container_width=True, theme=None)

    with st.expander("Qual a proporção de pessoas com e sem interações?", expanded=False):

        filtro_ativos_pizza = st.selectbox(
            "Quais pessoas incluir?",
            ["Todas as pessoas", "Apenas pessoas ativas hoje"],
            index=0,
            key="pizza_interacoes_ativos",
        )

        # 🔥 AGORA MULTISELECT
        tipo_pizza = st.multiselect(
            "Tipos de interação",
            options=["Mensagens", "Agendamentos", "Ligações", "Todas"],
            default=["Todas"],
            key="pizza_interacoes_tipo_multi",
        )

        # --------------------------------------------
        # INTERPRETAÇÃO DO MULTISELECT
        # --------------------------------------------
        if "Todas" in tipo_pizza:
            tipos_pizza = INTER_TIPOS
            titulo_pizza = "Pessoas com e sem interações (todas as interações)"
        else:
            tipos_pizza = []
            if "Mensagens" in tipo_pizza:
                tipos_pizza.extend(INTER_TIPOS_MS)
            if "Agendamentos" in tipo_pizza:
                tipos_pizza.extend(INTER_TIPOS_AT)
            if "Ligações" in tipo_pizza:
                tipos_pizza.extend(INTER_TIPOS_LG)

            titulo_pizza = (
                "Pessoas com e sem interações — " +
                ", ".join(tipo_pizza)
            )

        somente_ativos = (filtro_ativos_pizza == "Apenas pessoas ativas hoje")

        df_pizza = preparar_pizza_interacoes_zero_vs_maior_zero(
            dfF,
            tipos=tipos_pizza,
            somente_ativos_hoje=somente_ativos,
        )

        if df_pizza.empty or df_pizza["qtd"].sum() == 0:
            st.info("Não há pessoas no universo selecionado para montar o gráfico.")
        else:
            fig_pizza = pie_standard(
                df=df_pizza,
                names="grupo",
                values="qtd",
                title=titulo_pizza,
                hole=0.35,
                legend_pos="below_title",
                percent_digits=1,
                number_digits=0,
                min_pct_inside=5,
            )
            st.plotly_chart(fig_pizza, use_container_width=True)

    with st.expander("Como está distribuído o número de interações por pessoa?", expanded=False):

        filtro_ativos = st.selectbox(
            "Quais pessoas incluir?",
            ["Todas as pessoas", "Apenas pessoas ativas hoje"],
            index=0,
        )

        # 🔥 MULTISELECT AQUI
        tipo_hist = st.multiselect(
            "Tipos de interação",
            options=["Mensagens", "Agendamentos", "Ligações", "Todas"],
            default=["Todas"],
        )

        # --------------------------------------------
        # INTERPRETAÇÃO DO MULTISELECT
        # --------------------------------------------
        if "Todas" in tipo_hist:
            tipos = INTER_TIPOS
            titulo = "Distribuição de interações totais por pessoa"
        else:
            tipos = []
            if "Mensagens" in tipo_hist:
                tipos.extend(INTER_TIPOS_MS)
            if "Agendamentos" in tipo_hist:
                tipos.extend(INTER_TIPOS_AT)
            if "Ligações" in tipo_hist:
                tipos.extend(INTER_TIPOS_LG)

            titulo = (
                "Distribuição de interações por pessoa — "
                + ", ".join(tipo_hist)
            )

        # FILTROS DE INTERAÇÃO
        df_int = dfF[dfF["_tipo_norm"].isin(tipos)].copy()

        # FILTRO DE ATIVOS
        if filtro_ativos == "Apenas pessoas ativas hoje":
            ultimo_mes = dfF["_month_period"].max()
            ids_ativos = (
                dfF[dfF["_month_period"] == ultimo_mes]["id_pessoa"]
                .dropna()
                .unique()
            )
            df_int = df_int[df_int["id_pessoa"].isin(ids_ativos)]

        if df_int.empty:
            st.info("Não há interações para o tipo e filtro selecionados.")
        else:
            df_counts = (
                df_int.groupby("id_pessoa")
                .size()
                .reset_index(name="n_interacoes")
                .query("n_interacoes > 0")
            )

            N = len(df_counts)
            nbins = int(np.ceil(1 + np.log2(N)))

            fig = hist_interacoes_por_pessoa(
                df=df_counts,
                col_numerica="n_interacoes",
                nbins=nbins,
                titulo=titulo,
            )
            st.plotly_chart(fig, use_container_width=True)