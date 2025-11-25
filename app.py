# ---------------------------------------------------------------
# Set up
# ---------------------------------------------------------------
import io 
import re
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pandas as pd
import json, unicodedata
import os
from datetime import datetime
import pytz
import plotly.express as px
import hashlib

# ---------------------------------------------------------------
# Config da página
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="📊 Public Health Analytics")

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"  

def first_existing(*relative_paths: str) -> Path | None:
    for rel in relative_paths:
        p = ASSETS / rel
        if p.exists():
            return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Ana Health</h1>
        <p style='color: white;'>Explore os painéis para tomada de decisão baseada em dados</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* Esconde a lista padrão de páginas no topo da sidebar */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

def safe_page_link(path: str, label: str, icon: str | None = None):
    try:
        if (APP_DIR / path).exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página não disponível neste app.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    else:
        st.warning(f"Logo não encontrada em {ASSETS}/logo.(png|jpg|jpeg|webp)")
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    with st.expander("Painel Estratégico", expanded=False):
        safe_page_link("pages/relatorio_executivo.py", label="Relatório Executivo", icon="🎯")

    with st.expander("Painel Tático", expanded=False):
        safe_page_link("pages/relatorio_gestor.py", label="Relatório Gestor", icon="📈")

    with st.expander("Painel Operacional", expanded=False):
        safe_page_link("pages/relatorio_operacao.py", label="Operação", icon="🏨")


with st.sidebar:
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
# Leitura de CSV (upload)
# =========================
@st.cache_data(show_spinner=False)
def _read_csv_smart(file, force_sep: str | None = None, dtype_map: dict | None = None) -> pd.DataFrame:
    """
    Lê CSV/TXT detectando separador quando possível.
    - force_sep: se informado, usa explicitamente (ex.: ';').
    - dtype_map: map de dtypes, ex.: {'id_pessoa':'string'}
    """
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

# =========================
# Área principal (Landing)
# =========================

st.subheader("🧭 Sobre este painel")
st.write(
    """
Bem-vindo ao **Public Health Analytics**. Aqui você encontra três visões para explorar e comunicar dados em saúde:

- **Painel Executivo:** visão executiva com KPIs para macrogestores.
- **Painel Tático:** visão tática de metas, variações e alavancas operacionais.
- **Painel Operacional:** granularidade para ações do dia a dia.

Escolha abaixo por onde começar 👇
"""
)

def card(title: str, desc: str, icon: str, page_path: str):
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

        page_file = (APP_DIR / page_path)
        try:
            if page_file.exists():
                st.page_link(page_path, label=f"Abrir {title}", icon=icon, help=f"Ir para {title}")
            else:
                st.button(f"Abrir {title}", icon=icon, disabled=True, help="Página não disponível neste app.")
        except Exception:
            st.button(f"Abrir {title}", icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

c1, c2, c3 = st.columns(3)
with c1:
    card(
        title="Relatório Executivo",
        desc="Visão estratégica para analisar KPIs definidos.",
        icon="🎯",
        page_path="pages/relatorio_executivo.py",
    )
with c2:
    card(
        title="Relatório Gestor",
        desc="KPIs de gestão: metas, tendência, variação e explicações acionáveis.",
        icon="📈",
        page_path="pages/relatorio_gestor.py",
    )
with c3:
    card(
        title="Operação",
        desc="Detalhamento operacional por unidade, profissional e procedimento.",
        icon="🏨",
        page_path="pages/relatorio_operacao.py",
    )

st.divider()
st.info(
    "Dica: a qualquer momento, use o menu lateral para navegar. "
)