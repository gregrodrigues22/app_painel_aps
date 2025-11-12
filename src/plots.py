# src/plots.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from itertools import cycle
import unicodedata
from typing import Iterable, Optional

# ------------------------------------------------------------------
# Utilitário simples de formatação (sem separador de milhares)
# ------------------------------------------------------------------

def _fmt(v: float | int | None, thousands: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    try:
        v_int = int(round(float(v)))
    except Exception:
        return ""
    # por padrão NÃO usar separador de milhar
    return f"{v_int:,}".replace(",", ".") if thousands else f"{v_int}"

def _dias_uteis(start: pd.Timestamp,
                end: pd.Timestamp,
                feriados: Optional[Iterable[pd.Timestamp]] = None) -> pd.DatetimeIndex:
    """
    Retorna os dias úteis entre start e end, excluindo feriados.
    Aceita feriados como lista/iterável, DatetimeIndex, Series etc.
    """
    # normaliza o iterável de feriados sem usar 'or' (evita ambiguidade)
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


# ------------------------------------------------------------
# Utilitário simples de formatação numérica
# ------------------------------------------------------------
def _fmt(v: float | int | None, thousands: bool = False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return ""
    if thousands:
        # usa separador de milhar com vírgula e troca por ponto
        return f"{int(round(v)):,}".replace(",", ".")
    return str(int(round(v)))

# ------------------------------------------------------------------
# 1) Número de ativos por mês (pessoas distintas / mês) date_col deve ser datetime-like
# ------------------------------------------------------------------
def fig_ativos_por_mes(
    df: pd.DataFrame,
    *,
    date_col: str,       # coluna de data (datetime ou string YYYY-MM)
    id_col: str,         # id da pessoa (distinct/unique por mês)
    title: str = "Pessoas ativas por mês",
    trend_alpha: float = 0.10,
) -> go.Figure:
    """
    Recebe base detalhada (linhas transacionais) e produz a série
    mensal de pessoas ativas distintas.
    """
    d = df.copy()

    # mês como período e etiqueta "YYYY_QQ" / "YYYY-MM"
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
# Barras por período + YoY + média + tendência + (opcional) MM
# ------------------------------------------------------------------
def bar_yoy_trend(
    df: pd.DataFrame,
    *,
    x: str,                         # coluna do eixo X (mês/ano em string ou categoria)
    y: str,                         # métrica
    title: str = "Série temporal",
    x_is_year: bool = False,        # se True, ordena/complete anos (min..max)
    fill_missing_years: bool = True,
    show_ma: bool = True,
    ma_window: int = 3,
    show_mean: bool = True,
    show_trend: bool = True,
    trend_alpha: float = 0.10,      # suavização EWMA (0.05..0.3 fica bom)
    legend_pos: str = "top",        # "top" ou "bottom"
    y_label: str | None = None,
    thousands: bool = False,        # apenas mantém compat, não divide por mil
    x_angle: int = 90,              # rotação do rótulo do eixo X
) -> go.Figure:
    """
    Barras por período com:
      - acima/abaixo da média;
      - melhor/pior mês destacados;
      - média móvel, linha de tendência suavizada e linha de média;
      - rótulos MoM coloridos por ponto (verde/vermelho) e legenda com média do MoM.
    """
    # ---------- validações básicas ----------
    if df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Sem dados",
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=50, b=40),
        )
        return fig

    # ---------- prepara base ----------
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
        # mantém ordem original de entrada (ou já ordenada antes)
        d0[x] = d0[x].astype(str)
        d = d0.copy()
        d["x_str"] = d[x]
        cat_array = d["x_str"].tolist()

    # ---------- estatísticas ----------
    media = float(d[y].mean(skipna=True)) if show_mean else np.nan
    span = max(1.0, np.nanmax(d[y]) if d[y].notna().any() else 1.0)
    top_room = (np.nanmax(d[y]) if d[y].notna().any() else 1.0) + span * 0.12

    # melhor/pior
    best_idx = int(d[y].idxmax()) if d[y].notna().any() else None
    worst_idx = int(d[y].idxmin()) if d[y].notna().any() else None

    is_best = d.index == best_idx
    is_worst = d.index == worst_idx
    is_above = d[y] >= media if show_mean else pd.Series(False, index=d.index)
    is_below = d[y] <  media if show_mean else pd.Series(False, index=d.index)

    # ---------- figure ----------
    fig = go.Figure()

    # Helpers
    def _fmt(v, use_thousands=False):
        if pd.isna(v):
            return ""
        if use_thousands:
            return f"{float(v)/1000:,.1f}k".replace(",", ".")
        return f"{int(v):,}".replace(",", ".")

    # Paleta
    GREEN_LIGHT = "#a5d6a7"
    GREEN_STRONG = "#2e7d32"
    RED_LIGHT   = "#ef9a9a"
    RED_STRONG  = "#c62828"
    MEAN_COLOR  = "#7a7a7a"
    TREND_COLOR = "#6e6e6e"
    MA_COLOR    = "#1f77b4"

    # ---------- barras principais ----------
    # acima da média (exclui best/worst para evitar duplicação visual)
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

    # abaixo da média (exclui best/worst)
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

    # melhor mês
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

    # pior mês
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

    # ---------- média móvel ----------
    if show_ma:
        ma = d[y].rolling(ma_window, min_periods=1).mean()
        fig.add_scatter(
            x=d["x_str"],
            y=ma,
            mode="lines",
            line=dict(dash="dash", width=2, color=MA_COLOR),
            name=f"Média móvel ({ma_window})",
        )

    # ---------- tendência suavizada ----------
    if show_trend:
        trend = d[y].ewm(alpha=trend_alpha, adjust=False).mean()
        fig.add_scatter(
            x=d["x_str"],
            y=trend,
            mode="lines",
            line=dict(width=2.5, color=TREND_COLOR),
            name="Tendência suavizada",
        )

    # ---------- linha da média ----------
    if show_mean:
        fig.add_scatter(
            x=d["x_str"],
            y=[media] * len(d),
            mode="lines",
            line=dict(dash="dot", width=2, color=MEAN_COLOR),
            name=f"Média ({_fmt(media, thousands)})",
        )

    # ---------- Δ vs mês anterior (MoM) – rótulos coloridos ----------
    d["mom_pct"] = d[y].pct_change() * 100.0
    y_base = d[y].fillna(media if show_mean else d[y].median())
    offset = span * 0.06
    y_txt = y_base + offset

    mask_pos  = d["mom_pct"] > 0
    mask_neg  = d["mom_pct"] < 0
    mask_zero = d["mom_pct"].fillna(0).eq(0)

    def _fmt_pct(v):
        return "" if pd.isna(v) else f"{v:+.0f}%"

    # positivos
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
    # negativos
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
    # zeros/NaN (opcional)
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

    # legenda com média do MoM (dummy trace)
    mom_mean = float(np.nanmean(d["mom_pct"])) if d["mom_pct"].notna().any() else 0.0
    fig.add_scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#6e6e6e"),
        name=f"Δ vs mês anterior (MoM) — média: {mom_mean:+.1f}%",
        showlegend=True,
    )

    # ---------- layout e eixos ----------
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
        tickangle=x_angle,           # <<< rotação
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

def tau_trimestral_acumulado_por_dia(
    df: pd.DataFrame,
    *,
    date_col: str,                 # "data" | "mes_atendimento" | "_dt"
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
    dc = date_col if date_col in df.columns else (
        "_dt" if "_dt" in df.columns else
        "mes_atendimento" if "mes_atendimento" in df.columns else
        "data" if "data" in df.columns else None
    )
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

    # --------- normalização mínima ---------
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

    # --------- denominadores (ativos por trimestre) ---------
    ativos_q = d.groupby("_quarter")[id_col].nunique()

    # --------- primeira interação no trimestre ---------
    d_inter = d[d["_tipo_norm"].isin(inter_tipos)]
    first_inter = (
        d_inter.groupby(["_quarter", id_col])["_date"]
        .min()
        .reset_index(name="first_day")
    )

    # --------- gera a série diária acumulada de TAU por trimestre ---------
    palette_cycle = cycle(["#1E88E5", "#1E88E5"])
    series = []   # lista de dicts: {"q": Period, "df": df_tau, "color": color}

    for q, df_q in first_inter.groupby("_quarter"):
        denom = int(ativos_q.get(q, 0))
        if denom <= 0:
            continue

        q_start = q.start_time.normalize()
        q_end   = q.end_time.normalize()

        # contagem por dia do "primeiro engajamento" e cumulativa
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

    # média geral sobre todos os pontos de tau
    full = pd.concat([s["df"] for s in series], ignore_index=True)
    media_tau = float(full["tau"].mean())

    # --------- figura ---------
    fig = go.Figure()

    for s in series:
        qlbl = str(s["q"])            # ex.: '2023Q2'
        color = s["color"]
        df_tau = s["df"]

        # linha do trimestre
        fig.add_scatter(
            x=df_tau["date"], y=df_tau["tau"],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TAU: %{y:.2%}<extra></extra>",
            name=qlbl,
        )

        # rótulo com o % no ponto final (mesma cor da linha)
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

        # rótulo do trimestre no ponto de TAU máximo
        imax = int(np.nanargmax(df_tau["tau"].to_numpy()))
        x_max = df_tau["date"].iloc[imax]
        y_max = df_tau["tau"].iloc[imax]
        fig.add_annotation(
            x=x_max, y=y_max,
            text=qlbl,
            showarrow=False,
            yshift=14,
            font=dict(size=11, color="#555"),
            bgcolor="rgba(255,255,255,0.6)",
            align="center",
        )

    # linha da média geral
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
        showlegend=False,  # << remove legenda
    )
    return fig

def heatmap_combos_por_trimestre_colscale(
    df: pd.DataFrame,
    *,
    date_col: str,
    id_col: str = "id_pessoa",
    tipo_col: str = "tipo",
    relative: bool = False,   # escala do rótulo (o mapa de cores continua normalizado por coluna)
    title: str = "Engajamento por combinações de canais por trimestre (pessoas)",
    nenhum_last: bool = True,
) -> go.Figure:
    import unicodedata
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # ---------- colunas mínimas ----------
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

    # ---------- pivot absoluto (N de pessoas) ----------
    pivot_abs = (base
                 .groupby(["combo", "quarter_label"])[id_col]
                 .nunique()
                 .unstack("quarter_label", fill_value=0)
                 .sort_index(axis=1))

    if pivot_abs.empty:
        fig = go.Figure()
        fig.update_layout(title="Sem dados", template="plotly_white", height=320)
        return fig

    # ---------- matriz relativa (% por trimestre) ----------
    col_sums = pivot_abs.sum(axis=0).replace(0, np.nan)
    pivot_pct = (pivot_abs / col_sums) * 100.0

    # Ordenação pelo último trimestre (pela métrica exibida)
    last_col = pivot_abs.columns[-1]
    ser_last = (pivot_pct if relative else pivot_abs)[last_col].fillna(0)
    if nenhum_last and "nenhum" in ser_last.index:
        ordered = ser_last.drop(index="nenhum").sort_values(ascending=False).index.tolist() + ["nenhum"]
    else:
        ordered = ser_last.sort_values(ascending=False).index.tolist()

    pivot_abs = pivot_abs.loc[ordered]
    pivot_pct = pivot_pct.loc[ordered]

    # ---------- z (cores) normalizado por COLUNA da métrica escolhida ----------
    to_color = (pivot_pct if relative else pivot_abs).astype(float)
    z_norm = to_color.copy()
    for c in z_norm.columns:
        cmax = z_norm[c].max()
        z_norm[c] = 0.0 if (pd.isna(cmax) or cmax <= 0) else (z_norm[c] / cmax)

    # ---------- rótulos na célula & hover ----------
    if relative:
        # texto grande = porcentagem (inteiro), hover = % + N
        text_to_show = pivot_pct.values.astype(float)
        texttemplate = "%{text:.0f}%"
        hovertemplate = (
            "<b>%{y}</b><br>Trimestre: %{x}<br>"
            "Percentual: %{customdata[0]:.1f}%<br>"
            "Pessoas: %{customdata[1]:,d}<extra></extra>"
        )
    else:
        # texto grande = N, hover = N + %
        text_to_show = pivot_abs.values.astype(float)
        texttemplate = "%{text:.0f}"
        hovertemplate = (
            "<b>%{y}</b><br>Trimestre: %{x}<br>"
            "Pessoas: %{customdata[1]:,d}<br>"
            "Percentual: %{customdata[0]:.1f}%<extra></extra>"
        )

    # customdata carrega (pct, abs) para o hover
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
# Gráfico: TAU diário acumulado x Meta rateada por dia (cumul.)
# ------------------------------------------------------------
def meta_trimestral_acumulada(
    df: pd.DataFrame,
    *,
    date_col: str,                 # coluna de datas (diárias)
    id_col: str = "id_pessoa",     # identificador da pessoa
    tipo_col: str = "tipo",        # coluna do tipo de interação
    inter_tipos: Optional[set] = None,   # quais tipos contam como "engajamento"
    quarter_label: str,            # "YYYYQX" (aceita também "YYYYQ0X")
    meta_tau: float,               # meta do trimestre (0–1). Ex: 0.80 (80%)
    feriados: Optional[Iterable[str]] = None,  # lista opcional de feriados
    line_real="#1976d2",
    line_meta="#1b5e20",
    title="04 - TAU Realizado Acumulado e 09 - Meta Rateada por Dia Acumulada por date",
    return_kpis: bool = False      # <<< NOVO: se True, retorna (fig, kpis)
):
    import unicodedata, re
    if inter_tipos is None:
        inter_tipos = {
            "mensagem","mensagens","ligacao","ligações","ligacoes",
            "atendimento","consulta","whatsapp","msg","chamada"
        }

    # --- datas e quarter ---
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["_quarter"] = df[date_col].dt.to_period("Q")
    df["_quarter_lbl"] = df["_quarter"].astype(str)  # "YYYYQ1..4"

    # aceita 'YYYYQ01..04' e 'YYYYQ1..4'
    quarter_label_norm = re.sub(r"Q0([1-4])$", r"Q\1", str(quarter_label))

    if quarter_label_norm not in set(df["_quarter_lbl"]):
        fig = go.Figure()
        fig.update_layout(title=f"Sem dados para {quarter_label}", template="plotly_white", height=340)
        return (fig, {}) if return_kpis else fig

    # fatia o trimestre
    q_df = df[df["_quarter_lbl"] == quarter_label_norm].copy()

    # datas do tri
    q_period = q_df["_quarter"].iloc[0]
    q_start = q_period.start_time.normalize()
    q_end   = q_period.end_time.normalize()

    # --- denominador (ativos no trimestre) ---
    denom = int(q_df[id_col].nunique())
    if denom <= 0:
        fig = go.Figure()
        fig.update_layout(title="Sem denominador (ativos) no trimestre selecionado",
                          template="plotly_white", height=340)
        return (fig, {}) if return_kpis else fig

    # --- primeiro dia com interação por pessoa (dentro do tri) ---
    def _norm(s):
        s = str(s).strip().lower()
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch)!="Mn")
        return s
    if tipo_col in q_df.columns:
        q_df["_tipo_norm"] = q_df[tipo_col].map(_norm)
    else:
        q_df["_tipo_norm"] = ""

    eng = q_df[q_df["_tipo_norm"].isin(inter_tipos)].copy()
    eng["_date"] = eng[date_col].dt.normalize()

    first_inter = (
        eng.groupby(id_col)["_date"]
        .min()
        .rename("first_day")
        .reset_index()
    )

    # contagem diária de primeiras interações
    counts = first_inter["first_day"].value_counts().sort_index()
    # eixo diário completo do trimestre
    idx = pd.date_range(q_start, q_end, freq="D")
    cum_real = counts.reindex(idx, fill_value=0).cumsum() / denom

    # --- meta rateada por dia útil ---
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

    dias_uteis = _dias_uteis(q_start, q_end, feriados)
    if len(dias_uteis) == 0:
        dias_uteis = idx

    meta_dia = meta_tau / len(dias_uteis)
    meta_cum = pd.Series(0.0, index=idx)
    meta_cum.loc[dias_uteis] = meta_dia
    meta_cum = meta_cum.cumsum().clip(upper=meta_tau)

    # --- df final para o plot ---
    out = pd.DataFrame({
        "date": idx,
        "realizado_cum": cum_real.values,
        "meta_cum": meta_cum.values
    })
    out["gap"] = out["realizado_cum"] - out["meta_cum"]

    # KPIs (no último dia DISPONÍVEL nos dados de engajamento)
    if len(first_inter):
        last_data_day = min(first_inter["first_day"].max(), q_end)
    else:
        last_data_day = q_start - pd.Timedelta(days=1)  # nenhum engajado

    # engajados até a última data com dado
    if last_data_day < q_start:
        eng_ate_ultima_data = 0
        tau_ate_ultima_data = 0.0
    else:
        eng_ate_ultima_data = int((first_inter["first_day"] <= last_data_day).sum())
        tau_ate_ultima_data = eng_ate_ultima_data / denom

    # faltantes pra atingir a meta
    alvo_abs = int(np.ceil(meta_tau * denom))
    faltam_abs = max(0, alvo_abs - eng_ate_ultima_data)

    # dias úteis restantes após a última data com dado
    if last_data_day < q_end:
        d0 = (last_data_day + pd.Timedelta(days=1)).normalize()
        dias_restantes = _dias_uteis(d0, q_end, feriados)
        n_dias_rest = int(len(dias_restantes))
    else:
        n_dias_rest = 0

    faltam_por_dia = int(np.ceil(faltam_abs / max(1, n_dias_rest))) if faltam_abs > 0 else 0

    # --- Plot ---
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

    # anotações compactas
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

    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="04 - TAU Realizado Acumulado e 09 - Meta")
    fig.update_xaxes(title="date", showgrid=True)
    fig.update_layout(
        title=title + f" — {quarter_label_norm}  |  Ativos: {denom:,}  |  Meta alvo: {meta_tau:.0%}  |  Restante p/ meta: {restante_meta:.1%}",
        template="plotly_white", hovermode="x unified", height=420,
        margin=dict(l=40, r=10, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
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