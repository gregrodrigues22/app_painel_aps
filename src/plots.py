# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from itertools import cycle
import unicodedata
from typing import Iterable, Optional
from scipy import stats
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------

def _fmt(v: float | int | None, thousands: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    try:
        v_int = int(round(float(v)))
    except Exception:
        return ""
    return f"{v_int:,}".replace(",", ".") if thousands else f"{v_int}"

def _dias_uteis(start: pd.Timestamp,
                end: pd.Timestamp,
                feriados: Optional[Iterable[pd.Timestamp]] = None) -> pd.DatetimeIndex:
    """
    Retorna os dias úteis entre start e end, excluindo feriados.
    Aceita feriados como lista/iterável, DatetimeIndex, Series etc.
    """
    if feriados is None:
        feriados_list = []
    else:
        feriados_list = list(feriados)

    feriados_dt = pd.to_datetime(feriados_list, errors="coerce")
    # normaliza e remove NaT
    feriados_set = set([ts.normalize() for ts in feriados_dt if pd.notna(ts)])

    # índice diário completo
    idx = pd.date_range(start, end, freq="D")

    # dia útil = segunda–sexta e não está no conjunto de feriados
    mask_uteis = (idx.weekday < 5) & (~idx.normalize().isin(feriados_set))
    return idx[mask_uteis]

# ---------------------------
# Funções auxiliares
# ---------------------------
DEFAULT_COND_COLS = [
    "higido", "multicomorbido", "diabetes",
    "dislipidemia", "hipertensao", "obesidade", "saude_mental"
]

def detect_cond_cols(df: pd.DataFrame, cond_cols: list[str] | None = None) -> list[str]:
    """Retorna apenas as colunas de condição existentes no DataFrame."""
    base = cond_cols or DEFAULT_COND_COLS
    return [c for c in base if c in df.columns]

# ------------------------------------------------------------------
# Número de ativos por mês 
# ------------------------------------------------------------------
def fig_ativos_por_mes(
    df: pd.DataFrame,
    *,
    date_col: str,       
    id_col: str,       
    title: str = "Pessoas ativas por mês",
    trend_alpha: float = 0.10,
) -> go.Figure:
    """
    Recebe base detalhada (linhas transacionais) e produz a série
    mensal de pessoas ativas distintas.
    """
    d = df.copy()

    dt = pd.to_datetime(d[date_col], errors="coerce")
    d["_period"] = dt.dt.to_period("M").astype(str)

    agg = (
        d.groupby("_period")[id_col]
        .nunique()
        .reset_index()
        .rename(columns={"_period": "periodo", id_col: "ativos"})
    )

    return bar_yoy_trend(
        agg,
        x="periodo",
        y="ativos",
        title=title,
        x_is_year=False,
        fill_missing_years=False,
        show_ma=True,
        ma_window=3,
        show_mean=True,
        show_trend=True,
        legend_pos="top",
        y_label="Pessoas",
        thousands=False,
        trend_alpha=trend_alpha,
    )

# ------------------------------------------------------------------
# Barras por período + YoY + média + tendência 
# ------------------------------------------------------------------
def bar_yoy_trend(
    df: pd.DataFrame,
    *,
    x: str,                    
    y: str,                      
    title: str = "Série temporal",
    x_is_year: bool = False,      
    fill_missing_years: bool = True,
    show_ma: bool = True,
    ma_window: int = 3,
    show_mean: bool = True,
    show_trend: bool = True,
    trend_alpha: float = 0.10,     
    legend_pos: str = "top",     
    y_label: str | None = None,
    thousands: bool = False,      
    x_angle: int = 90,            
) -> go.Figure:
    """
    Barras por período com:
      - acima/abaixo da média;
      - melhor/pior mês destacados;
      - média móvel, linha de tendência suavizada e linha de média;
      - rótulos MoM coloridos por ponto (verde/vermelho) e legenda com média do MoM.
    """

    if df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Sem dados",
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=50, b=40),
        )
        return fig

    d0 = df[[x, y]].copy()
    d0[y] = pd.to_numeric(d0[y], errors="coerce")

    if x_is_year:
        d0[x] = pd.to_numeric(d0[x], errors="coerce").astype("Int64")
        d0 = d0.sort_values(x)
        if fill_missing_years and d0[x].notna().any():
            anos = pd.Series(range(int(d0[x].min()), int(d0[x].max()) + 1), name=x)
            d = anos.to_frame().merge(d0, on=x, how="left")
        else:
            d = d0.copy()
        d["x_str"] = d[x].astype(str)
        cat_array = d["x_str"].tolist()
    else:
        d0[x] = d0[x].astype(str)
        d = d0.copy()
        d["x_str"] = d[x]
        cat_array = d["x_str"].tolist()

    media = float(d[y].mean(skipna=True)) if show_mean else np.nan
    span = max(1.0, np.nanmax(d[y]) if d[y].notna().any() else 1.0)
    top_room = (np.nanmax(d[y]) if d[y].notna().any() else 1.0) + span * 0.12

    best_idx = int(d[y].idxmax()) if d[y].notna().any() else None
    worst_idx = int(d[y].idxmin()) if d[y].notna().any() else None

    is_best = d.index == best_idx
    is_worst = d.index == worst_idx
    is_above = d[y] >= media if show_mean else pd.Series(False, index=d.index)
    is_below = d[y] <  media if show_mean else pd.Series(False, index=d.index)

    fig = go.Figure()

    def _fmt(v, use_thousands=False):
        if pd.isna(v):
            return ""
        if use_thousands:
            return f"{float(v)/1000:,.1f}k".replace(",", ".")
        return f"{int(v):,}".replace(",", ".")

    GREEN_LIGHT = "#a5d6a7"
    GREEN_STRONG = "#2e7d32"
    RED_LIGHT   = "#ef9a9a"
    RED_STRONG  = "#c62828"
    MEAN_COLOR  = "#7a7a7a"
    TREND_COLOR = "#6e6e6e"
    MA_COLOR    = "#1f77b4"

    mask_up = is_above & ~is_best & ~is_worst
    if mask_up.any():
        fig.add_bar(
            x=d.loc[mask_up, "x_str"],
            y=d.loc[mask_up, y],
            name="Acima da média",
            marker_color=GREEN_LIGHT,
            marker_line_color="#455A64",
            marker_line_width=0.6,
            text=[_fmt(v, thousands) for v in d.loc[mask_up, y]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="black", size=11),
            hovertemplate="<b>%{x}</b><br>Valor: %{y:,}<extra></extra>",
        )

    mask_dn = is_below & ~is_best & ~is_worst
    if mask_dn.any():
        fig.add_bar(
            x=d.loc[mask_dn, "x_str"],
            y=d.loc[mask_dn, y],
            name="Abaixo da média",
            marker_color=RED_LIGHT,
            marker_line_color="#455A64",
            marker_line_width=0.6,
            text=[_fmt(v, thousands) for v in d.loc[mask_dn, y]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="black", size=11),
            hovertemplate="<b>%{x}</b><br>Valor: %{y:,}<extra></extra>",
        )

    if is_best.any():
        fig.add_bar(
            x=d.loc[is_best, "x_str"],
            y=d.loc[is_best, y],
            name="Melhor mês",
            marker_color=GREEN_STRONG,
            marker_line_color="#2b3d2b",
            marker_line_width=0.8,
            text=[_fmt(v, thousands) for v in d.loc[is_best, y]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{x}</b><br>Valor: %{y:,}<extra></extra>",
        )

    if is_worst.any():
        fig.add_bar(
            x=d.loc[is_worst, "x_str"],
            y=d.loc[is_worst, y],
            name="Pior mês",
            marker_color=RED_STRONG,
            marker_line_color="#3d1f1f",
            marker_line_width=0.8,
            text=[_fmt(v, thousands) for v in d.loc[is_worst, y]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{x}</b><br>Valor: %{y:,}<extra></extra>",
        )

    if show_ma:
        ma = d[y].rolling(ma_window, min_periods=1).mean()
        fig.add_scatter(
            x=d["x_str"],
            y=ma,
            mode="lines",
            line=dict(dash="dash", width=2, color=MA_COLOR),
            name=f"Média móvel ({ma_window})",
        )

    if show_trend:
        trend = d[y].ewm(alpha=trend_alpha, adjust=False).mean()
        fig.add_scatter(
            x=d["x_str"],
            y=trend,
            mode="lines",
            line=dict(width=2.5, color=TREND_COLOR),
            name="Tendência suavizada",
        )

    if show_mean:
        fig.add_scatter(
            x=d["x_str"],
            y=[media] * len(d),
            mode="lines",
            line=dict(dash="dot", width=2, color=MEAN_COLOR),
            name=f"Média ({_fmt(media, thousands)})",
        )

    d["mom_pct"] = d[y].pct_change() * 100.0
    y_base = d[y].fillna(media if show_mean else d[y].median())
    offset = span * 0.06
    y_txt = y_base + offset

    mask_pos  = d["mom_pct"] > 0
    mask_neg  = d["mom_pct"] < 0
    mask_zero = d["mom_pct"].fillna(0).eq(0)

    def _fmt_pct(v):
        return "" if pd.isna(v) else f"{v:+.0f}%"

    fig.add_scatter(
        x=d.loc[mask_pos, "x_str"],
        y=y_txt.loc[mask_pos],
        mode="text",
        text=[_fmt_pct(v) for v in d.loc[mask_pos, "mom_pct"]],
        textfont=dict(color=GREEN_STRONG, size=11),
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    fig.add_scatter(
        x=d.loc[mask_neg, "x_str"],
        y=y_txt.loc[mask_neg],
        mode="text",
        text=[_fmt_pct(v) for v in d.loc[mask_neg, "mom_pct"]],
        textfont=dict(color=RED_STRONG, size=11),
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    fig.add_scatter(
        x=d.loc[mask_zero, "x_str"],
        y=y_txt.loc[mask_zero],
        mode="text",
        text=[_fmt_pct(v) for v in d.loc[mask_zero, "mom_pct"]],
        textfont=dict(color="#9e9e9e", size=11),
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    mom_mean = float(np.nanmean(d["mom_pct"])) if d["mom_pct"].notna().any() else 0.0
    fig.add_scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#6e6e6e"),
        name=f"Δ vs mês anterior (MoM) — média: {mom_mean:+.1f}%",
        showlegend=True,
    )

    if legend_pos == "bottom":
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0)
        top_margin = 60
        title_pad_b = 6
    else:
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0)
        top_margin = 110
        title_pad_b = 28

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        rangemode="tozero",
        range=[0, top_room],
        title=y_label or "Pessoas",
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=cat_array,
        tickmode="array",
        tickvals=cat_array,
        ticktext=cat_array,
        tickangle=x_angle,         
    )
    fig.update_layout(
        title=dict(text=title, y=0.98, pad=dict(b=title_pad_b)),
        barmode="overlay",
        bargap=0.02,
        bargroupgap=0.0,
        hovermode="x unified",
        legend=legend_cfg,
        margin=dict(l=60, r=30, t=top_margin, b=70),
        paper_bgcolor="white",
    )

    return fig

# ------------------------------------------------------------------
# TAU Trimestral Acumulado por DIA
# ------------------------------------------------------------------
def tau_trimestral_acumulado_por_dia(
    df: pd.DataFrame,
    *,
    date_col: str,                
    id_col: str = "id_pessoa",
    tipo_col: str = "tipo",
    inter_tipos: set | None = None,
    title: str = "TAU Realizado Acumulado por date",
    mean_color: str = "#6b6b6b",
) -> go.Figure:

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados", template="plotly_white", height=320)
        return fig

    # --------- colunas realmente existentes ---------
    possiveis = [date_col, "data", "_dt", "mes_atendimento"]
    dc = next((c for c in possiveis if c in df.columns), None)

    if dc is None:
        raise KeyError("Coluna de data não encontrada (tente _dt, mes_atendimento ou data).")

    if id_col not in df.columns:
        raise KeyError(f"Coluna de id '{id_col}' não encontrada.")

    tc = tipo_col
    if tc not in df.columns:
        for cand in ["tipo_interacao", "interaction_type", "evento", "tipo"]:
            if cand in df.columns:
                tc = cand
                break
        else:
            raise KeyError(f"Coluna de tipo '{tipo_col}' não encontrada e nenhum alias comum foi achado.")

    d = df[[dc, id_col, tc]].copy()

    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d = d.dropna(subset=[dc])
    d["_date"] = d[dc].dt.normalize()
    d["_quarter"] = d["_date"].dt.to_period("Q")

    if not inter_tipos:
        inter_tipos = {
            "mensagem", "mensagens", "msg", "whatsapp",
            "ligacao", "ligacoes", "chamada",
            "atendimento", "consulta"
        }

    import unicodedata
    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
        return s

    d["_tipo_norm"] = d[tc].map(_norm)

    ativos_q = d.groupby("_quarter")[id_col].nunique()

    d_inter = d[d["_tipo_norm"].isin(inter_tipos)]
    first_inter = (
        d_inter.groupby(["_quarter", id_col])["_date"]
        .min()
        .reset_index(name="first_day")
    )

    palette_cycle = cycle(["#1E88E5", "#1E88E5"])
    series = [] 

    for q, df_q in first_inter.groupby("_quarter"):
        denom = int(ativos_q.get(q, 0))
        if denom <= 0:
            continue

        q_start = q.start_time.normalize()
        q_end   = q.end_time.normalize()

        s_counts = df_q["first_day"].value_counts().sort_index()
        idx = pd.date_range(q_start, q_end, freq="D")
        cum_engajados = s_counts.reindex(idx, fill_value=0).cumsum()
        tau = cum_engajados / denom

        df_tau = pd.DataFrame({"date": idx, "tau": tau.values})
        series.append({"q": q, "df": df_tau, "color": next(palette_cycle)})

    if not series:
        fig = go.Figure()
        fig.update_layout(title="Sem dados (sem trimestres com denominador válido).",
                          template="plotly_white", height=320)
        return fig

    full = pd.concat([s["df"] for s in series], ignore_index=True)
    media_tau = float(full["tau"].mean())

    fig = go.Figure()

    for s in series:
        qlbl = str(s["q"])        
        color = s["color"]
        df_tau = s["df"]

        fig.add_scatter(
            x=df_tau["date"], y=df_tau["tau"],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TAU: %{y:.2%}<extra></extra>",
            name=qlbl,
        )

        x_end = df_tau["date"].iloc[-1]
        y_end = df_tau["tau"].iloc[-1]
        fig.add_scatter(
            x=[x_end], y=[y_end],
            mode="text",
            text=[f"{y_end:.0%}"],
            textposition="top center",
            textfont=dict(size=11, color=color),
            hoverinfo="skip",
            showlegend=False,
        )

        imax = int(np.nanargmax(df_tau["tau"].to_numpy()))
        x_max = df_tau["date"].iloc[imax]
        y_max = df_tau["tau"].iloc[imax]
        fig.add_annotation(
            x=x_max,
            y=min(y_max - 0.05, 0.95),  
            text=qlbl,
            showarrow=False,
            yshift=0,
            font=dict(size=11, color="#555"),
            bgcolor="rgba(255,255,255,0.6)",
            align="center",
        )

    fig.add_scatter(
        x=[full["date"].min(), full["date"].max()],
        y=[media_tau, media_tau], mode="lines",
        line=dict(color=mean_color, width=2, dash="dot"),
        name=f"Média ({media_tau:.2%})",
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="TAU Realizado Acumulado")
    fig.update_xaxes(title="date", showgrid=True)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x",
        margin=dict(l=40, r=10, t=60, b=40),
        showlegend=False, 
    )
    return fig

# ------------------------------------------------------------------
# TAU Mensal por Trimestre em Heatmap
# ------------------------------------------------------------------
def tau_distribuicao_mensal_por_trimestre(
    df: pd.DataFrame,
    *,
    date_col: str,               
    id_col: str = "id_pessoa",
    tipo_col: str = "tipo",
    inter_tipos: set | None = None,
    title: str = "Distribuição do engajamento dentro do trimestre",
) -> go.Figure:

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados", template="plotly_white", height=320)
        return fig

    possiveis = [date_col, "data", "_dt", "mes_atendimento"]
    dc = next((c for c in possiveis if c in df.columns), None)

    if dc is None:
        raise KeyError(
            f"Coluna de data não encontrada. Recebi date_col='{date_col}'."
        )

    if id_col not in df.columns:
        raise KeyError(f"Coluna de id '{id_col}' não encontrada.")

    tc = tipo_col
    if tc not in df.columns:
        for cand in ["tipo_interacao", "interaction_type", "evento", "tipo"]:
            if cand in df.columns:
                tc = cand
                break
        else:
            raise KeyError(
                f"Coluna de tipo '{tipo_col}' não encontrada e nenhum alias comum foi achado."
            )

    d = df[[dc, id_col, tc]].copy()

    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d = d.dropna(subset=[dc])
    d["_date"] = d[dc].dt.normalize()
    d["_quarter"] = d["_date"].dt.to_period("Q")

    if not inter_tipos:
        inter_tipos = {
            "mensagem", "mensagens", "msg", "whatsapp",
            "ligacao", "ligacoes", "chamada",
            "atendimento", "consulta"
        }

    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(
            ch for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )
        return s

    d["_tipo_norm"] = d[tc].map(_norm)

    d_inter = d[d["_tipo_norm"].isin(inter_tipos)]
    if d_inter.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Sem dados de engajamento (nenhuma interação nos tipos especificados).",
            template="plotly_white",
            height=320,
        )
        return fig

    first_inter = (
        d_inter.groupby(["_quarter", id_col])["_date"]
        .min()
        .reset_index(name="first_day")
    )

    first_inter["_q_start_month"] = first_inter["_quarter"].dt.start_time.dt.month
    first_inter["_month_in_quarter"] = (
        first_inter["first_day"].dt.month - first_inter["_q_start_month"] + 1
    )

    first_inter = first_inter[
        first_inter["_month_in_quarter"].isin([1, 2, 3])
    ].copy()

    dist = (
        first_inter
        .groupby(["_quarter", "_month_in_quarter"])[id_col]
        .nunique()
        .reset_index(name="engajados")
    )

    if dist.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem engajados por trimestre.", template="plotly_white", height=320)
        return fig

    quarters = dist["_quarter"].unique()
    idx = pd.MultiIndex.from_product(
        [quarters, [1, 2, 3]],
        names=["_quarter", "_month_in_quarter"],
    )

    dist = (
        dist.set_index(["_quarter", "_month_in_quarter"])
            .reindex(idx, fill_value=0)
            .reset_index()
    )

    dist["total_q"] = dist.groupby("_quarter")["engajados"].transform("sum")
    dist = dist[dist["total_q"] > 0].copy()
    dist["prop"] = dist["engajados"] / dist["total_q"]

    dist["_quarter_str"] = dist["_quarter"].astype(str)
    month_labels = {1: "1º mês", 2: "2º mês", 3: "3º mês"}
    dist["month_label"] = dist["_month_in_quarter"].map(month_labels)

    heat = (
        dist
        .pivot(index="month_label", columns="_quarter_str", values="prop")
        .sort_index() 
    )

    text_vals = heat.applymap(lambda v: f"{v:.0%}" if pd.notna(v) else "")

    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=list(heat.columns),
            y=list(heat.index),
            zmin=0,
            zmax=1,
            colorscale="Blues",
            colorbar=dict(title="Proporção do engajamento", tickformat=".0%"),
            text=text_vals.values,
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Trimestre",
        yaxis_title="Mês dentro do trimestre",
        margin=dict(l=60, r=20, t=60, b=60),
    )

    return fig
    
# ------------------------------------------------------------------
# TAU Mensal por Trimestre em Barras
# ------------------------------------------------------------------
def tau_distribuicao_mensal_por_trimestre_barras(
    df: pd.DataFrame,
    *,
    date_col: str,                
    id_col: str = "id_pessoa",
    tipo_col: str = "tipo",
    inter_tipos: set | None = None,
    title: str = "Distribuição do engajamento por mês do trimestre",
) -> go.Figure:

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados", template="plotly_white", height=320)
        return fig

    possiveis = [date_col, "data", "_dt", "mes_atendimento"]
    dc = next((c for c in possiveis if c in df.columns), None)

    if dc is None:
        raise KeyError(
            f"Coluna de data não encontrada. Recebi date_col='{date_col}'."
        )

    if id_col not in df.columns:
        raise KeyError(f"Coluna de id '{id_col}' não encontrada.")

    tc = tipo_col
    if tc not in df.columns:
        for cand in ["tipo_interacao", "interaction_type", "evento", "tipo"]:
            if cand in df.columns:
                tc = cand
                break
        else:
            raise KeyError(
                f"Coluna de tipo '{tipo_col}' não encontrada e nenhum alias comum foi achado."
            )

    d = df[[dc, id_col, tc]].copy()

    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d = d.dropna(subset=[dc])
    d["_date"] = d[dc].dt.normalize()
    d["_quarter"] = d["_date"].dt.to_period("Q")

    if not inter_tipos:
        inter_tipos = {
            "mensagem", "mensagens", "msg", "whatsapp",
            "ligacao", "ligacoes", "chamada",
            "atendimento", "consulta"
        }

    import unicodedata
    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(
            ch for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )
        return s

    d["_tipo_norm"] = d[tc].map(_norm)

    d_inter = d[d["_tipo_norm"].isin(inter_tipos)]
    if d_inter.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Sem dados de engajamento (nenhuma interação nos tipos especificados).",
            template="plotly_white",
            height=320,
        )
        return fig

    first_inter = (
        d_inter.groupby(["_quarter", id_col])["_date"]
        .min()
        .reset_index(name="first_day")
    )

    first_inter["_q_start_month"] = first_inter["_quarter"].dt.start_time.dt.month
    first_inter["_month_in_quarter"] = (
        first_inter["first_day"].dt.month - first_inter["_q_start_month"] + 1
    )

    first_inter = first_inter[
        first_inter["_month_in_quarter"].isin([1, 2, 3])
    ].copy()

    dist = (
        first_inter
        .groupby(["_quarter", "_month_in_quarter"])[id_col]
        .nunique()
        .reset_index(name="engajados")
    )

    if dist.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem engajados por trimestre.", template="plotly_white", height=320)
        return fig

    quarters = sorted(dist["_quarter"].unique())
    idx = pd.MultiIndex.from_product(
        [quarters, [1, 2, 3]],
        names=["_quarter", "_month_in_quarter"],
    )

    dist = (
        dist.set_index(["_quarter", "_month_in_quarter"])
            .reindex(idx, fill_value=0)
            .reset_index()
    )

    dist["total_q"] = dist.groupby("_quarter")["engajados"].transform("sum")
    dist = dist[dist["total_q"] > 0].copy()
    dist["prop"] = dist["engajados"] / dist["total_q"]

    month_labels = {1: "1º mês", 2: "2º mês", 3: "3º mês"}
    quarter_labels = [str(q) for q in quarters]

    color_map = {
        1: "#1E88E5", 
        2: "#64B5F6",  
        3: "#BBDEFB",  
    }

    fig = go.Figure()
    for m in [1, 2, 3]:
        sub = (
            dist[dist["_month_in_quarter"] == m]
            .set_index("_quarter")
            .reindex(quarters, fill_value=0)
        )

        labels = [
            f"{p:.0%}" if p >= 0.08 else ""
            for p in sub["prop"].values
        ]

        fig.add_bar(
            x=quarter_labels,
            y=sub["prop"].values,
            name=month_labels[m],
            marker_color=color_map[m],
            text=labels,
            textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=(
                "Trimestre: %{x}<br>"
                f"Mês: {month_labels[m]}<br>"
                "Percentual do engajamento: %{y:.0%}<extra></extra>"
            ),
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        template="plotly_white",
        xaxis_title="Trimestre",
        yaxis_title="Proporção do engajamento no trimestre",
        yaxis=dict(tickformat=".0%"),
        legend_title_text="Mês dentro do trimestre",
        margin=dict(l=60, r=20, t=60, b=60),
    )

    return fig

# ------------------------------------------------------------------
# Heatmap Tipo de Engajamento
# ------------------------------------------------------------------
def heatmap_combos_por_trimestre_colscale(
    df: pd.DataFrame,
    *,
    date_col: str,
    id_col: str = "id_pessoa",
    tipo_col: str = "tipo",
    relative: bool = False,  
    title: str = "Engajamento por combinações de canais por trimestre (pessoas)",
    nenhum_last: bool = True,
) -> go.Figure:
    import unicodedata
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if date_col not in df.columns:
        raise KeyError(f"Coluna de data '{date_col}' não encontrada no DataFrame.")
    if id_col not in df.columns:
        raise KeyError(f"Coluna id '{id_col}' não encontrada no DataFrame.")
    if tipo_col not in df.columns:
        raise KeyError(f"Coluna tipo '{tipo_col}' não encontrada no DataFrame.")

    d = df[[date_col, id_col, tipo_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["_quarter"] = d[date_col].dt.to_period("Q")
    d["quarter_label"] = d["_quarter"].astype(str)

    def _norm_text(s):
        s = str(s).strip().lower()
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
        return s

    d["_tipo_norm"] = d[tipo_col].map(_norm_text)

    map_mensagem    = {"mensagem", "mensagens", "msg", "whatsapp"}
    map_ligacao     = {"ligacao", "ligacoes", "chamada", "call", "telefone", "phone"}
    map_atendimento = {"atendimento", "atendimentos", "consulta", "consultas", "agenda"}

    d["is_mensagem"]    = d["_tipo_norm"].isin(map_mensagem).astype(int)
    d["is_ligacao"]     = d["_tipo_norm"].isin(map_ligacao).astype(int)
    d["is_atendimento"] = d["_tipo_norm"].isin(map_atendimento).astype(int)
    d["ativo"] = 1

    agg = (d
           .groupby([id_col, "quarter_label"], as_index=False)
           .agg({
                "ativo": "max",
                "is_mensagem": "max",
                "is_ligacao": "max",
                "is_atendimento": "max"
           }))

    order_pref = ["mensagem", "ligacao", "atendimentos"]
    def make_combo(row) -> str:
        ativos = []
        if row["is_mensagem"] > 0:
            ativos.append("mensagem")
        if row["is_ligacao"] > 0:
            ativos.append("ligacao")
        if row["is_atendimento"] > 0:
            ativos.append("atendimentos")
        if not ativos:
            return "nenhum"
        ativos = sorted(ativos, key=lambda x: order_pref.index(x))
        return "+".join(ativos)

    agg["combo"] = agg.apply(make_combo, axis=1)
    base = agg.loc[agg["ativo"] == 1, ["quarter_label", "combo", id_col]].copy()

    pivot_abs = (base
                 .groupby(["combo", "quarter_label"])[id_col]
                 .nunique()
                 .unstack("quarter_label", fill_value=0)
                 .sort_index(axis=1))

    if pivot_abs.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados", template="plotly_white", height=320)
        return fig

    col_sums = pivot_abs.sum(axis=0).replace(0, np.nan)
    pivot_pct = (pivot_abs / col_sums) * 100.0

    last_col = pivot_abs.columns[-1]
    ser_last = (pivot_pct if relative else pivot_abs)[last_col].fillna(0)
    if nenhum_last and "nenhum" in ser_last.index:
        ordered = ser_last.drop(index="nenhum").sort_values(ascending=False).index.tolist() + ["nenhum"]
    else:
        ordered = ser_last.sort_values(ascending=False).index.tolist()

    pivot_abs = pivot_abs.loc[ordered]
    pivot_pct = pivot_pct.loc[ordered]

    to_color = (pivot_pct if relative else pivot_abs).astype(float)
    z_norm = to_color.copy()
    for c in z_norm.columns:
        cmax = z_norm[c].max()
        z_norm[c] = 0.0 if (pd.isna(cmax) or cmax <= 0) else (z_norm[c] / cmax)

    if relative:
        text_to_show = pivot_pct.values.astype(float)
        texttemplate = "%{text:.0f}%"
        hovertemplate = (
            "<b>%{y}</b><br>Trimestre: %{x}<br>"
            "Percentual: %{customdata[0]:.1f}%<br>"
            "Pessoas: %{customdata[1]:,d}<extra></extra>"
        )
    else:
        text_to_show = pivot_abs.values.astype(float)
        texttemplate = "%{text:.0f}"
        hovertemplate = (
            "<b>%{y}</b><br>Trimestre: %{x}<br>"
            "Pessoas: %{customdata[1]:,d}<br>"
            "Percentual: %{customdata[0]:.1f}%<extra></extra>"
        )

    customdata = np.dstack([pivot_pct.values, pivot_abs.values]).reshape(pivot_abs.shape[0], pivot_abs.shape[1], 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_norm.values,
            x=z_norm.columns.tolist(),
            y=z_norm.index.tolist(),
            text=text_to_show,
            texttemplate=texttemplate,
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            showscale=True,
            colorbar=dict(title="Intensidade (coluna)")
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=40),
        xaxis=dict(title="Trimestre", type="category"),
        yaxis=dict(title="Forma de engajamento no trimestre", autorange="reversed"),
    )
    return fig

# ------------------------------------------------------------
#  TAU diário acumulado x Meta rateada por dia (cumul.)
# ------------------------------------------------------------
def meta_trimestral_acumulada(
    df: pd.DataFrame,
    *,
    date_col: str,                
    id_col: str = "id_pessoa",     
    tipo_col: str = "tipo",        
    inter_tipos: Optional[set] = None,   
    quarter_label: str,            
    meta_tau: float,              
    feriados: Optional[Iterable[str]] = None,  
    line_real="#1976d2",
    line_meta="#1b5e20",
    title="04 - TAU Realizado Acumulado e 09 - Meta Rateada por Dia Acumulada por date",
    return_kpis: bool = False     
):
    import unicodedata, re

    if inter_tipos is None:
        inter_tipos = {
            "mensagem","mensagens","ligacao","ligações","ligacoes",
            "atendimento","consulta","whatsapp","msg","chamada"
        }

    possiveis = [date_col, "data", "_dt", "mes_atendimento"]
    dc = next((c for c in possiveis if c in df.columns), None)
    if dc is None:
        raise KeyError(
            f"Nenhuma coluna de data encontrada. Recebi date_col='{date_col}'."
        )

    df = df.copy()
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df = df.dropna(subset=[dc])

    df["_quarter"] = df[dc].dt.to_period("Q")
    df["_quarter_lbl"] = df["_quarter"].astype(str)

    quarter_label_norm = re.sub(r"Q0([1-4])$", r"Q\1", str(quarter_label))

    if quarter_label_norm not in set(df["_quarter_lbl"]):
        fig = go.Figure()
        fig.update_layout(
            title=f"Sem dados para {quarter_label}",
            template="plotly_white",
            height=340
        )
        return (fig, {}) if return_kpis else fig

    q_df = df[df["_quarter_lbl"] == quarter_label_norm].copy()

    q_period = q_df["_quarter"].iloc[0]
    q_start = q_period.start_time.normalize()
    q_end   = q_period.end_time.normalize()

    denom = int(q_df[id_col].nunique())
    if denom <= 0:
        fig = go.Figure()
        fig.update_layout(
            title="Sem denominador (ativos) no trimestre selecionado",
            template="plotly_white",
            height=340
        )
        return (fig, {}) if return_kpis else fig

    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(
            ch for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )
        return s

    if tipo_col in q_df.columns:
        q_df["_tipo_norm"] = q_df[tipo_col].map(_norm)
    else:
        q_df["_tipo_norm"] = ""

    eng = q_df[q_df["_tipo_norm"].isin(inter_tipos)].copy()
    eng["_date"] = eng[dc].dt.normalize()

    first_inter = (
        eng.groupby(id_col)["_date"]
        .min()
        .rename("first_day")
        .reset_index()
    )

    counts = first_inter["first_day"].value_counts().sort_index()
    idx = pd.date_range(q_start, q_end, freq="D")
    cum_real = counts.reindex(idx, fill_value=0).cumsum() / denom

    if feriados is None:
        feriados = pd.DatetimeIndex([])
    else:
        feriados = pd.to_datetime(list(feriados), errors="coerce")
        feriados = feriados[~feriados.isna()]

    def _dias_uteis(start, end, fer=None):
        if fer is None:
            fer_set = set()
        else:
            fer_set = set(pd.to_datetime(list(fer)))
        idx = pd.date_range(start, end, freq="D")
        mask = (idx.weekday < 5) & (~idx.isin(fer_set))
        return idx[mask]

    dias_uteis = _dias_uteis(q_start, q_end, feriados)
    if len(dias_uteis) == 0:
        dias_uteis = idx

    meta_dia = meta_tau / len(dias_uteis)
    meta_cum = pd.Series(0.0, index=idx)
    meta_cum.loc[dias_uteis] = meta_dia
    meta_cum = meta_cum.cumsum().clip(upper=meta_tau)

    out = pd.DataFrame({
        "date": idx,
        "realizado_cum": cum_real.values,
        "meta_cum": meta_cum.values
    })
    out["gap"] = out["realizado_cum"] - out["meta_cum"]

    if len(first_inter):
        last_data_day = min(first_inter["first_day"].max(), q_end)
    else:
        last_data_day = q_start - pd.Timedelta(days=1)  

    if last_data_day < q_start:
        eng_ate_ultima_data = 0
        tau_ate_ultima_data = 0.0
    else:
        eng_ate_ultima_data = int((first_inter["first_day"] <= last_data_day).sum())
        tau_ate_ultima_data = eng_ate_ultima_data / denom

    alvo_abs = int(np.ceil(meta_tau * denom))
    faltam_abs = max(0, alvo_abs - eng_ate_ultima_data)

    if last_data_day < q_end:
        d0 = (last_data_day + pd.Timedelta(days=1)).normalize()
        dias_restantes = _dias_uteis(d0, q_end, feriados)
        n_dias_rest = int(len(dias_restantes))
    else:
        n_dias_rest = 0

    faltam_por_dia = int(np.ceil(faltam_abs / max(1, n_dias_rest))) if faltam_abs > 0 else 0

    last = out.iloc[-1]
    realizado_final = float(last["realizado_cum"])
    meta_final      = float(last["meta_cum"])
    restante_meta   = max(0.0, meta_tau - realizado_final)

    fig = go.Figure()
    fig.add_scatter(
        x=out["date"], y=out["realizado_cum"],
        mode="lines", line=dict(width=3, color=line_real),
        name="04 - TAU Realizado Acumulado",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Realizado: %{y:.2%}<extra></extra>",
        fill="tozeroy", opacity=0.25
    )
    fig.add_scatter(
        x=out["date"], y=out["meta_cum"],
        mode="lines", line=dict(width=2, color=line_meta),
        name="09 - Meta Rateada por Dia Acumulada",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Meta (Acum.): %{y:.2%}<extra></extra>",
        fill=None
    )

    fig.add_annotation(
        x=out["date"].iloc[-1], y=realizado_final,
        xanchor="right", yanchor="bottom",
        text=f"<b>Realizado: {realizado_final:.1%}</b>",
        showarrow=False, font=dict(color=line_real)
    )
    fig.add_annotation(
        x=out["date"].iloc[-1], y=meta_final,
        xanchor="right", yanchor="top",
        text=f"<b>Meta acum.: {meta_final:.1%}</b>",
        showarrow=False, font=dict(color=line_meta)
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        title="04 - TAU Realizado Acumulado e 09 - Meta"
    )
    fig.update_xaxes(title="date", showgrid=True)
    fig.update_layout(
        title=(
            title
            + f" — {quarter_label_norm}  |  Ativos: {denom:,}  |  Meta alvo: {meta_tau:.0%}  |  Restante p/ meta: {restante_meta:.1%}"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=10, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    kpis = {
        "ativos": denom,
        "ultima_data": pd.to_datetime(last_data_day).date() if last_data_day >= q_start else None,
        "engajados_ate_ultima_data": eng_ate_ultima_data,
        "tau_ate_ultima_data": tau_ate_ultima_data,     # 0–1
        "alvo_definido_tau": meta_tau,                  # 0–1
        "faltam_abs": faltam_abs,                       # pessoas
        "dias_uteis_restantes": n_dias_rest,
        "faltam_por_dia": faltam_por_dia                # pessoas/dia útil
    }

    return (fig, kpis) if return_kpis else fig

# ------------------------------------------------------------
# Gráfico Setor
# ------------------------------------------------------------
def pie_standard(
    df: pd.DataFrame,
    names: str,               
    values: str,              
    title: str = "",
    hole: float = 0.35,        
    sort: str = "desc",        
    top_n: int | None = None,  
    others_label: str = "Outros",
    none_label: str = "(não informado)",
    show_legend: bool = True,
    legend_pos: str = "below_title",  
    percent_digits: int = 1,
    number_digits: int = 0,
    thousands_sep: str = ".",
    min_pct_inside: float = 6,
    color_discrete_map: dict | None = None,
):
    import re
    import plotly.express as px

    if (
        df is None or df.empty
        or names not in df.columns
        or values not in df.columns
    ):
        return px.pie(
            pd.DataFrame({names: [], values: []}),
            names=names, values=values, title=title, hole=hole
        )

    d = df[[names, values]].copy()
    d[names] = d[names].fillna(none_label).astype(str)
    d[values] = pd.to_numeric(d[values], errors="coerce").fillna(0)
    d = d.groupby(names, as_index=False)[values].sum()

    if top_n is not None and top_n > 0 and len(d) > top_n:
        d = d.sort_values(values, ascending=False)
        top = d.head(top_n)
        tail = d.iloc[top_n:]
        outros_val = tail[values].sum()
        if outros_val > 0:
            d = pd.concat(
                [top, pd.DataFrame({names: [others_label], values: [outros_val]})],
                ignore_index=True
            )

    if sort == "desc":
        d = d.sort_values(values, ascending=False)
    elif sort == "asc":
        d = d.sort_values(values, ascending=True)

    total = d[values].sum()
    if total == 0:
        return px.pie(
            pd.DataFrame({names: [], values: []}),
            names=names, values=values, title=title, hole=hole
        )

    d["pct"] = (d[values] / total * 100)

    def fmt_num(x: float) -> str:
        s = f"{x:,.{number_digits}f}"
        return s.replace(",", "X").replace(".", thousands_sep).replace("X", ",")

    d["label_inside"] = d.apply(
        lambda r: f"{r['pct']:.{percent_digits}f}%\n{fmt_num(r[values])}",
        axis=1
    )

    textpos = ["inside" if p >= min_pct_inside else "outside" for p in d["pct"]]
    pull = [0.0 if p >= min_pct_inside else 0.03 for p in d["pct"]]

    fig = px.pie(
        d,
        names=names,
        values=values,
        title=title,
        hole=hole,
        color=names,
        color_discrete_map=color_discrete_map,
    )

    val_fmt = f",.{number_digits}f"
    pct_fmt = f".{percent_digits}%"

    fig.update_traces(
        text=d["label_inside"],
        textposition=textpos,
        textinfo="none",
        texttemplate="%{text}",
        pull=pull,
        hovertemplate=(
            "<b>%{label}</b><br>"
            f"Quantidade: %{{value:{val_fmt}}}<br>"
            f"Participação: %{{percent:{pct_fmt}}}"
            "<extra></extra>"
        ),
        textfont=dict(size=12),
        sort=False,
    )

    if legend_pos == "below_title":
        legend_cfg = dict(
            orientation="h",
            yanchor="top",
            y=1.0,             
            xanchor="right",
            x=0.0,
            valign="top"
        )
        title_pad = 90         
    elif legend_pos == "bottom":
        legend_cfg = dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
        title_pad = 60
    else:
        legend_cfg = dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
        title_pad = 60

    fig.update_layout(
        showlegend=show_legend,
        legend=legend_cfg,
        margin=dict(l=10, r=10, t=title_pad, b=10),
        uniformtext_minsize=10,
        uniformtext_mode="show",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#f7f9fb",
    )

    return fig

# ------------------------------------------------------------
# Gráfico Histograma e Boxplot
# ------------------------------------------------------------

def histograma_boxplot_idade_plotly(
    df,
    col_numerica="col_numerica",       
    nbins=20,                           
    largura_px=1200,                     
    altura_px=800,                      
    margem=dict(l=300, r=140, t=100, b=60), 

    titulo="Distribuição de Idades dos Alunos",
):
    """
    Histograma + Curva Normal + Boxplot (subplots)
    Linha 1 (75%): histograma + curva Normal + média/mediana.
    Linha 2 (25%): boxplot horizontal compartilhando o eixo X.
    """

    # Paleta
    cor_histograma = "rgb(19, 93, 171)"
    cor_box        = "rgba(19, 93, 171, 0.6)"
    cor_outlier    = "rgba(19, 93, 171, 1.0)"
    cor_borda      = "white"
    cor_media      = "black"
    cor_mediana    = "green"
    cor_normal     = "red"
    cor_fence      = "rgba(0,0,0,0.35)"
    grid_color     = "rgba(0,0,0,0.08)"

    if col_numerica not in df.columns:
        raise ValueError(f"A coluna '{col_numerica}' não existe no DataFrame.")

    serie = pd.to_numeric(df[col_numerica], errors="coerce").dropna().astype(float)
    if serie.empty:
        raise ValueError(
            f"A coluna '{col_numerica}' não possui valores numéricos válidos."
        )

    total = int(len(serie))

    media    = float(np.mean(serie))
    mediana  = float(np.median(serie))
    sd       = float(np.std(serie, ddof=0))
    minimo   = float(serie.min())
    maximo   = float(serie.max())

    q1, q3   = np.percentile(serie, [25, 75])
    iqr      = float(q3 - q1)
    fence_low  = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr

    n_out_low  = int((serie < fence_low).sum())
    n_out_high = int((serie > fence_high).sum())

    NBINS = int(max(1, nbins))
    counts, edges = np.histogram(serie, bins=NBINS)
    bin_w = np.diff(edges)
    bin_c = (edges[:-1] + edges[1:]) / 2
    bin_l = edges[:-1]
    bin_r = edges[1:]
    perc  = (counts / total) * 100.0

    txt = [
        f"{c:,.0f}<br>({p:.1f}%)".replace(",", ".")
        for c, p in zip(counts, perc)
    ]

    customdata = np.column_stack([bin_l, bin_r, perc / 100.0])

    x_vals = np.linspace(serie.min(), serie.max(), 200)
    sd_eff = sd if sd > 0 else 1e-9
    y_vals = stats.norm.pdf(x_vals, loc=media, scale=sd_eff) * total * bin_w.mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Bar(
            x=bin_c,
            y=counts,
            width=bin_w,
            name="Distribuição",
            marker=dict(color=cor_histograma, line=dict(width=1, color=cor_borda)),
            text=txt,
            textposition="outside",
            textfont=dict(size=11),
            customdata=customdata,
            hovertemplate=(
                "Faixa do bin: %{customdata[0]:.1f} – %{customdata[1]:.1f}<br>"
                "Contagem: %{y:,}<br>"
                "Percentual: %{customdata[2]:.1%}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="Curva Normal",
            line=dict(color=cor_normal),
        ),
        row=1,
        col=1,
    )

    ymax_hist = max(
        max(counts) if len(counts) else 0,
        float(np.max(y_vals)) if len(y_vals) else 0,
    )

    fig.add_trace(
        go.Scatter(
            x=[media, media],
            y=[0, ymax_hist],
            mode="lines",
            name=f"Média: {media:.2f}",
            line=dict(color=cor_media, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[mediana, mediana],
            y=[0, ymax_hist],
            mode="lines",
            name=f"Mediana: {mediana:.2f}",
            line=dict(color=cor_mediana, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Box(
            x=serie,
            name="Boxplot",
            orientation="h",
            boxpoints="outliers",
            marker=dict(color=cor_box, outliercolor=cor_outlier, opacity=0.7),
            line=dict(color=cor_borda),
            hovertemplate="Valor=%{x}<extra></extra>",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    fig.add_shape(
        type="line",
        x0=media,
        x1=media,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color=cor_media, dash="dash", width=2),
    )
    fig.add_shape(
        type="line",
        x0=mediana,
        x1=mediana,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color=cor_mediana, dash="dash", width=2),
    )

    fig.add_shape(
        type="line",
        x0=fence_low,
        x1=fence_low,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color=cor_fence, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=fence_high,
        x1=fence_high,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color=cor_fence, dash="dot"),
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=cor_fence, dash="dot"),
            name="Limites de outlier (IQR×1.5)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"Outliers abaixo: {n_out_low:,}".replace(",", "."),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"Outliers acima: {n_out_high:,}".replace(",", "."),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"Mínimo: {minimo:.2f}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"Máximo: {maximo:.2f}",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial", color="#1f2c56"),  # sem negrito
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=largura_px,
        height=altura_px,
        margin=margem,
        bargap=0.1,
        legend=dict(x=0.86, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )

    fig.update_yaxes(range=[0, ymax_hist * 1.25], row=1, col=1)

    fig.update_yaxes(
        title_text="Número de pessoas",
        showgrid=True,
        gridcolor=grid_color,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Idade",
        showgrid=True,
        gridcolor=grid_color,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Idade",
        showgrid=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Boxplot",
        showgrid=False,
        row=2,
        col=1,
    )

    return fig
