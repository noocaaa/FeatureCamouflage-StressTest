"""
dashboard.py  —  Feature Camouflage · Results Dashboard
python dashboard.py  →  http://127.0.0.1:8050
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_here        = Path(__file__).resolve().parent
PROJECT_ROOT = _here if (_here / "results").exists() else _here.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "results"

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#05070f"
SURF    = "#0b0f1c"
CARD    = "#0f1525"
BORDER  = "#1a2840"
BORDER2 = "#243554"
TEXT    = "#c9d8f0"
MUTED   = "#3d5470"
ACCENT  = "#3b82f6"
CYAN    = "#06b6d4"
PURPLE  = "#8b5cf6"
GREEN   = "#10b981"
RED     = "#ef4444"
AMBER   = "#f59e0b"

COLORS = {
    "LOF":"#f87171","IDK":"#fb923c","iForest":"#fbbf24",
    "ONE":"#a3e635","Radar":"#34d399",
    "DOMINANT":"#60a5fa","AnomalyDAE":"#a78bfa","DONE":"#f472b6",
    "CoLA":"#38bdf8","GCAD":"#818cf8","GCAD+IDK":"#c084fc",
    "GAE":"#94a3b8",
}
GC   = {"none": AMBER, "indirect": ACCENT, "direct": GREEN}
DSC  = {"inj_cora": ACCENT, "inj_amazon": "#f472b6", "weibo": AMBER, "reddit": GREEN}
LTD  = {"ONE","Radar"}
ORD  = ["LOF","IDK","iForest","ONE","Radar","DOMINANT","AnomalyDAE","DONE","CoLA","GCAD","GCAD+IDK","GAE"]

dc  = lambda d: COLORS.get(d,"#888")
acl = lambda df: "auc_mean" if "auc_mean" in df.columns else "auc"

def hex_rgba(h, a=0.2):
    """Convert #rrggbb to rgba() — required for older Plotly on Windows."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11, family="'JetBrains Mono',monospace"),
    margin=dict(l=52,r=18,t=44,b=46),
    legend=dict(bgcolor="rgba(15,21,37,0.95)", bordercolor=BORDER2,
                borderwidth=1, font=dict(color=TEXT,size=10)),
)

def AX(t="", tv=None, r=None):
    d = dict(gridcolor=BORDER, gridwidth=0.5, zerolinecolor=BORDER2,
             linecolor=BORDER2, tickfont=dict(color=MUTED,size=10),
             title=t, title_font=dict(color=MUTED,size=11))
    if tv is not None: d["tickvals"] = tv
    if r  is not None: d["range"]    = r
    return d

# ── Data ───────────────────────────────────────────────────────────────────────
def load(f):
    p = RESULTS/f
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

sbm    = load("sbm_results.csv")
raw    = load("sbm_raw.csv")
rel    = load("structure_reliance.csv")
cola   = load("cola_graph_vs_noedge.csv")
colafd = load("cola_feature_distance.csv")
real   = load("real_results.csv")

GAMMAS = sorted(sbm["gamma"].unique())    if not sbm.empty  else []
CONDS  = sorted(sbm["condition"].unique()) if not sbm.empty  else []
DETS   = [d for d in ORD if d in sbm["detector"].unique()]  if not sbm.empty  else []
RDETS  = [d for d in ORD if d in real["detector"].unique()] if not real.empty else []
RDSS   = sorted(real["dataset"].unique()) if not real.empty else []
RELDTS = sorted(rel["detector"].unique()) if not rel.empty  else []
GRP    = (sbm.drop_duplicates("detector").set_index("detector")["group"].to_dict()
          if not sbm.empty else {})

# ── UI helpers ─────────────────────────────────────────────────────────────────
def card(ch, s=None):
    base = {"backgroundColor":CARD,"borderRadius":"8px","padding":"16px",
            "border":f"1px solid {BORDER}"}
    if s: base.update(s)
    return html.Div(ch, style=base)

def stl(t):
    return html.Div(t, style={"color":MUTED,"fontSize":"9px","letterSpacing":"3px",
                               "textTransform":"uppercase","marginBottom":"8px","fontWeight":"600"})

def rad(id_, val, opts, inline=False):
    ls = {"display":"inline-flex" if inline else "flex","alignItems":"center",
          "marginRight":"14px" if inline else "0","marginBottom":"0" if inline else "6px",
          "cursor":"pointer","color":TEXT,"fontSize":"12px"}
    return dcc.RadioItems(id=id_, value=val,
        options=[{"label":html.Span(o[0] if isinstance(o,tuple) else o,
                                    style={"color":TEXT,"fontSize":"12px"}),
                  "value":o[1] if isinstance(o,tuple) else o} for o in opts],
        inputStyle={"marginRight":"6px","accentColor":ACCENT}, labelStyle=ls)

def chk(id_, val, opts):
    return dcc.Checklist(id=id_, value=val,
        options=[{"label":html.Span([
            html.Span("● ",style={"color":dc(o),"fontSize":"10px"}),
            html.Span(o,  style={"color":TEXT,"fontSize":"11px"}),
        ]),"value":o} for o in opts],
        inputStyle={"marginRight":"5px","accentColor":ACCENT},
        labelStyle={"display":"flex","alignItems":"center","marginBottom":"4px","cursor":"pointer"})

def drp(id_, val, opts, **kw):
    return dcc.Dropdown(id=id_, value=val,
        options=[{"label":o,"value":o} for o in opts],
        style={"backgroundColor":SURF,"color":TEXT,"border":f"1px solid {BORDER}",
               "fontFamily":"monospace","fontSize":"12px"}, **kw)

def kpi(label, value, color=ACCENT, sub=""):
    return html.Div([
        html.Div(label, style={"color":MUTED,"fontSize":"9px","letterSpacing":"2px",
                                "textTransform":"uppercase","marginBottom":"5px"}),
        html.Div(str(value), style={"color":color,"fontSize":"21px",
                                     "fontWeight":"700","lineHeight":"1"}),
        html.Div(sub, style={"color":MUTED,"fontSize":"9px","marginTop":"3px"}),
    ], style={"backgroundColor":SURF,"borderRadius":"6px","padding":"13px 15px",
              "border":f"1px solid {BORDER}","minWidth":"115px"})

def hln(fig, y, lbl="random", color=RED, **kw):
    fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1,
                  annotation_text=lbl, annotation_font_color=color,
                  annotation_font_size=9, annotation_position="bottom right", **kw)

TS = {"backgroundColor":SURF,"color":MUTED,"border":f"1px solid {BORDER}",
      "borderRadius":"6px 6px 0 0","padding":"9px 16px",
      "fontFamily":"monospace","fontSize":"11px","letterSpacing":"0.5px"}
TA = {**TS,"backgroundColor":CARD,"color":ACCENT,"borderBottom":f"2px solid {ACCENT}"}

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Camouflage · Results"

app.layout = html.Div(
    style={"backgroundColor":BG,"minHeight":"100vh",
           "fontFamily":"'JetBrains Mono',monospace","padding":"20px 24px"},
    children=[
        html.Link(rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"),

        html.Div([
            html.Div([
                html.Div("FEATURE CAMOUFLAGE IN GRAPH NETWORKS",
                         style={"color":MUTED,"fontSize":"9px","letterSpacing":"4px","marginBottom":"4px"}),
                html.H1("Anomaly Detection · Stress Test Dashboard",
                        style={"color":TEXT,"margin":"0","fontSize":"17px","fontWeight":"600"}),
            ]),
            html.Div([
                html.Span(f"{len(DETS)} detectors",
                          style={"color":ACCENT,"fontSize":"10px","backgroundColor":SURF,
                                 "padding":"4px 10px","borderRadius":"20px",
                                 "border":f"1px solid {BORDER}","marginRight":"8px"}),
                html.Span(f"{len(GAMMAS)} γ levels",
                          style={"color":GREEN,"fontSize":"10px","backgroundColor":SURF,
                                 "padding":"4px 10px","borderRadius":"20px",
                                 "border":f"1px solid {BORDER}","marginRight":"8px"}),
                html.Span(f"{len(CONDS)} conditions",
                          style={"color":PURPLE,"fontSize":"10px","backgroundColor":SURF,
                                 "padding":"4px 10px","borderRadius":"20px",
                                 "border":f"1px solid {BORDER}"}),
            ]),
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                  "marginBottom":"18px","paddingBottom":"16px","borderBottom":f"1px solid {BORDER}"}),

        dcc.Tabs(id="tabs", value="overview", style={"marginBottom":"16px"}, children=[
            dcc.Tab(label="Overview",      value="overview",   style=TS, selected_style=TA),
            dcc.Tab(label="AUC vs γ",      value="auc",        style=TS, selected_style=TA),
            dcc.Tab(label="Bump chart",    value="bump",       style=TS, selected_style=TA),
            dcc.Tab(label="Heatmap",       value="heatmap",    style=TS, selected_style=TA),
            dcc.Tab(label="Robustness",    value="robustness", style=TS, selected_style=TA),
            dcc.Tab(label="Seed variance", value="seeds",      style=TS, selected_style=TA),
            dcc.Tab(label="CoLA",          value="cola",       style=TS, selected_style=TA),
            dcc.Tab(label="Structure",     value="structure",  style=TS, selected_style=TA),
            dcc.Tab(label="Real datasets", value="real",       style=TS, selected_style=TA),
            dcc.Tab(label="SBM vs Real",   value="transfer",   style=TS, selected_style=TA),
        ]),
        html.Div(id="tab-content"),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURES (built once at tab render, no callbacks needed)
# ══════════════════════════════════════════════════════════════════════════════

def _kpis():
    col = acl(sbm)
    at0  = sbm[(sbm.gamma==0)&(sbm.condition=="A")&(sbm.camouflage=="global")]
    at1  = sbm[(sbm.gamma==1.0)&(sbm.condition=="A")&(sbm.camouflage=="global")]
    b0   = at0.loc[at0[col].idxmax(),"detector"] if not at0.empty else "—"
    b1   = at1.loc[at1[col].idxmax(),"detector"] if not at1.empty else "—"
    gae  = sbm[sbm.detector=="GAE"][col].mean()
    gcad = sbm[sbm.detector=="GCAD+IDK"][col].mean()
    seeds = int(raw["seed"].max())+1 if not raw.empty else 5
    rho = np.nan
    if not sbm.empty and not real.empty:
        sbm0  = sbm[(sbm.gamma==0)&(sbm.condition=="A")&(sbm.camouflage=="global")].set_index("detector")[col]
        rmu   = real.groupby("detector")["auc"].mean()
        common= [d for d in RDETS if d in sbm0.index]
        if len(common) > 2:
            rho, _ = stats.spearmanr([sbm0[d] for d in common],[rmu[d] for d in common])
    return html.Div([
        kpi("Best @ γ=0",      b0,            ACCENT,  "Cond A · global"),
        kpi("Best @ γ=1",      b1,            GREEN,   "Most robust to camouflage"),
        kpi("GAE mean AUC",    f"{gae:.3f}",  "#94a3b8","Structure-only baseline"),
        kpi("GCAD+IDK mean",   f"{gcad:.3f}", PURPLE,  "Full method"),
        kpi("Seeds / run",     seeds,         AMBER,   "per detector × γ"),
        kpi("SBM→Real ρ",      f"{rho:.2f}",  RED,     "Spearman rank correlation"),
    ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"16px"})


def _fig_group_ab():
    col = acl(sbm)
    fig = go.Figure()
    seen = set()
    for cond in ["A","B"]:
        for cam in ["global","local"]:
            sub = sbm[(sbm.condition==cond)&(sbm.camouflage==cam)]
            if sub.empty: continue
            for grp in ["none","indirect","direct"]:
                sg = sub[(sub.group==grp)&(~sub.detector.isin(LTD))].groupby("gamma")[col].mean().reset_index()
                if sg.empty: continue
                c   = GC[grp]
                lbl = f"{grp} · Cond {cond} · {cam}"
                dash_ = "solid" if cam=="global" else "dash"
                width_ = 2.5 if cond=="B" else 1.5
                showleg = lbl not in seen
                seen.add(lbl)
                fig.add_trace(go.Scatter(
                    x=sg.gamma, y=sg[col], mode="lines", name=lbl,
                    showlegend=showleg,
                    line=dict(color=c, width=width_, dash=dash_),
                    hovertemplate=f"<b>{lbl}</b><br>γ=%{{x:.1f}}<br>mean AUC=%{{y:.3f}}<extra></extra>"))
    hln(fig, 0.5)
    fig.update_layout(**BASE,
        title="Group mean AUC · Cond A (attribute) vs B (bridge) · solid=global · dash=local · thick=Cond B",
        xaxis=AX("γ",GAMMAS), yaxis=AX("mean AUC",r=[0,1.05]))
    return fig


def _fig_cond_c():
    col = acl(sbm)
    sub = sbm[(sbm.condition=="C")&(sbm.gamma==0.0)].set_index("detector")[col]
    order = [d for d in ORD if d in sub.index]
    vals  = [sub[d] for d in order]
    clrs  = [GC.get(GRP.get(d,"none"),"#888") for d in order]
    fig   = go.Figure()
    for det, v, c in zip(order, vals, clrs):
        fig.add_shape(type="line", x0=det, x1=det, y0=0.5, y1=v,
                      line=dict(color=c,width=2))
    fig.add_trace(go.Scatter(
        x=order, y=vals, mode="markers+text",
        marker=dict(color=clrs,size=12,line=dict(color=BG,width=2)),
        text=[f"{v:.3f}" for v in vals],
        textposition="top center", textfont=dict(color=TEXT,size=9),
        hovertemplate="<b>%{x}</b><br>AUC=%{y:.3f}<extra></extra>",
        showlegend=False))
    hln(fig, 0.5)
    hln(fig, 1.0, "perfect", GREEN)
    fig.update_layout(**BASE,
        title="Condition C · Clique (purely structural anomalies, no feature signal)",
        xaxis=AX(), yaxis=AX("AUC",r=[0,1.12]))
    return fig


def _fig_ab_gamma1():
    col = acl(sbm)
    a1 = sbm[(sbm.condition=="A")&(sbm.camouflage=="global")&(sbm.gamma==1.0)].set_index("detector")[col]
    b1 = sbm[(sbm.condition=="B")&(sbm.camouflage=="global")&(sbm.gamma==1.0)].set_index("detector")[col]
    order = [d for d in ORD if d in a1.index and d in b1.index]
    clrs  = [GC.get(GRP.get(d,"none"),"#888") for d in order]
    fig   = go.Figure()
    fig.add_trace(go.Bar(x=order, y=[a1[d] for d in order],
        name="Cond A @ γ=1 (feature-only, fully camouflaged)",
        marker_color=clrs, opacity=0.4, marker_line_width=0,
        hovertemplate="<b>%{x}</b> Cond A<br>AUC=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=order, y=[b1[d] for d in order],
        name="Cond B @ γ=1 (bridge, structural signal survives)",
        marker_color=clrs, opacity=1.0, marker_line_width=0,
        hovertemplate="<b>%{x}</b> Cond B<br>AUC=%{y:.3f}<extra></extra>"))
    hln(fig, 0.5)
    fig.update_layout(**BASE, barmode="group",
        title="AUC at full camouflage γ=1 · Cond A vs B · global  (bright bar = structural signal protects)",
        xaxis={**AX(), "tickangle":-35}, yaxis=AX("AUC @ γ=1",r=[0,1.05]))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB ROUTER
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render(tab):

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if tab == "overview":
        if sbm.empty:
            return html.P("sbm_results.csv not found. Run run_sbm.py first.",style={"color":MUTED})
        return html.Div([
            _kpis(),
            html.Div([
                card([dcc.Graph(figure=_fig_group_ab(), style={"height":"340px"})], s={"flex":"2"}),
                card([dcc.Graph(figure=_fig_cond_c(),   style={"height":"340px"})], s={"flex":"1"}),
            ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
            card([dcc.Graph(figure=_fig_ab_gamma1(), style={"height":"300px"})]),
        ])

    # ── AUC vs γ ──────────────────────────────────────────────────────────────
    elif tab == "auc":
        if sbm.empty: return html.P("sbm_results.csv not found.",style={"color":MUTED})
        ctrl = card([html.Div([
            html.Div([stl("Condition"),  rad("av-cond","A",["A","B","C"])],                                  style={"minWidth":"90px"}),
            html.Div([stl("Camouflage"), rad("av-cam","global",[("Global","global"),("Local","local"),("None","none")])],style={"minWidth":"140px"}),
            html.Div([stl("Metric"),     rad("av-met","auc",[("AUC-ROC","auc"),("Avg Precision","ap")])],    style={"minWidth":"140px"}),
            html.Div([stl("Bands"),      rad("av-bands","show",[("Show","show"),("Hide","hide")])],           style={"minWidth":"100px"}),
            html.Div([stl("Detectors"),  chk("av-dets",DETS,DETS)],                                          style={"flex":"1"}),
        ], style={"display":"flex","gap":"20px","flexWrap":"wrap"})],s={"marginBottom":"12px"})
        return html.Div([ctrl, card([dcc.Graph(id="av-chart", style={"height":"480px"})])])

    # ── BUMP CHART ────────────────────────────────────────────────────────────
    elif tab == "bump":
        if sbm.empty: return html.P("sbm_results.csv not found.",style={"color":MUTED})
        ctrl = card([html.Div([
            html.Div([stl("Condition"),  rad("bmp-cond","A",["A","B"])],                                      style={"minWidth":"90px"}),
            html.Div([stl("Camouflage"), rad("bmp-cam","global",[("Global","global"),("Local","local")])],     style={"minWidth":"140px"}),
        ], style={"display":"flex","gap":"20px"})],s={"marginBottom":"12px"})
        note = card([html.Div(
            "Each line = one detector's rank among all 12 as γ increases. "
            "Rank 1 = best AUC. Crossing lines reveal which detectors survive camouflage and which collapse.",
            style={"color":MUTED,"fontSize":"11px"})],
            s={"marginBottom":"12px","borderColor":AMBER+"44"})
        return html.Div([ctrl, note, card([dcc.Graph(id="bmp-chart", style={"height":"520px"})])])

    # ── HEATMAP ───────────────────────────────────────────────────────────────
    elif tab == "heatmap":
        if sbm.empty: return html.P("sbm_results.csv not found.",style={"color":MUTED})
        ctrl = card([html.Div([
            html.Div([stl("Condition"),  rad("hm-cond","A",["A","B","C"])],                                   style={"minWidth":"90px"}),
            html.Div([stl("Camouflage"), rad("hm-cam","global",[("Global","global"),("Local","local"),("None","none")])],style={"minWidth":"140px"}),
            html.Div([stl("Sort"),       rad("hm-sort","group",[("Group","group"),("AUC@0","auc0"),("Mean","mean")])],   style={"minWidth":"140px"}),
            html.Div([stl("Metric"),     rad("hm-met","auc",[("AUC-ROC","auc"),("Avg Precision","ap")])],     style={"minWidth":"140px"}),
        ], style={"display":"flex","gap":"20px","flexWrap":"wrap"})],s={"marginBottom":"12px"})
        return html.Div([ctrl, card([dcc.Graph(id="hm-chart", style={"height":"500px"})])])

    # ── ROBUSTNESS ────────────────────────────────────────────────────────────
    elif tab == "robustness":
        if sbm.empty: return html.P("sbm_results.csv not found.",style={"color":MUTED})
        col = acl(sbm)
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=["Δ AUC · Cond A · global","Δ AUC · Cond A · local",
                            "Δ AUC · Cond B · global vs local","AUC · Cond C (structural clique)"],
            vertical_spacing=0.18, horizontal_spacing=0.09)

        for (cond,cam),(row,c_) in zip([("A","global"),("A","local")],[(1,1),(1,2)]):
            sub = sbm[(sbm.condition==cond)&(sbm.camouflage==cam)]
            g0  = sub[sub.gamma==sub.gamma.min()].set_index("detector")[col]
            gm  = sub[sub.gamma==sub.gamma.max()].set_index("detector")[col]
            dlt = (g0-gm).reindex(ORD).dropna()
            clrs= [GC.get(GRP.get(d,"none"),"#888") for d in dlt.index]
            fig.add_trace(go.Bar(x=dlt.index, y=dlt.values, marker_color=clrs,
                marker_line_width=0, showlegend=False,
                hovertemplate="<b>%{x}</b><br>ΔAUC=%{y:.3f}<extra></extra>"), row=row, col=c_)
            fig.add_hline(y=0, line_color=MUTED, line_width=0.8, row=row, col=c_)

        for cam, op in [("global",1.0),("local",0.5)]:
            sub = sbm[(sbm.condition=="B")&(sbm.camouflage==cam)]
            g0  = sub[sub.gamma==sub.gamma.min()].set_index("detector")[col]
            gm  = sub[sub.gamma==sub.gamma.max()].set_index("detector")[col]
            dlt = (g0-gm).reindex(ORD).dropna()
            clrs= [GC.get(GRP.get(d,"none"),"#888") for d in dlt.index]
            fig.add_trace(go.Bar(x=dlt.index, y=dlt.values, marker_color=clrs, opacity=op,
                marker_line_width=0, name=f"B · {cam}",
                hovertemplate="<b>%{x}</b><br>ΔAUC=%{y:.3f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0, line_color=MUTED, line_width=0.8, row=2, col=1)

        sub_c = sbm[(sbm.condition=="C")&(sbm.gamma==0.0)].set_index("detector")[col]
        sub_c = sub_c.reindex(ORD).dropna()
        clrs_c= [GC.get(GRP.get(d,"none"),"#888") for d in sub_c.index]
        fig.add_trace(go.Bar(x=sub_c.index, y=sub_c.values, marker_color=clrs_c,
            marker_line_width=0, showlegend=False,
            hovertemplate="<b>%{x}</b><br>AUC=%{y:.3f}<extra></extra>"), row=2, col=2)
        fig.add_hline(y=0.5, line_dash="dot", line_color=RED, line_width=1, row=2, col=2)

        fig.update_layout(**BASE, height=640,
            title="Robustness across all conditions  (orange=none · blue=indirect · green=direct)")
        for r in [1,2]:
            for c_ in [1,2]:
                fig.update_xaxes(tickangle=-35, tickfont=dict(color=MUTED,size=9), row=r, col=c_)
                fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER2,
                                 tickfont=dict(color=MUTED,size=9), row=r, col=c_)
        return card([dcc.Graph(figure=fig, style={"height":"660px"})])

    # ── SEED VARIANCE ─────────────────────────────────────────────────────────
    elif tab == "seeds":
        if raw.empty: return html.P("sbm_raw.csv not found.",style={"color":MUTED})
        rdets = [d for d in ORD if d in raw["detector"].unique()]
        ctrl  = card([html.Div([
            html.Div([stl("Detector"),   drp("sd-det",rdets[0],rdets)],                                       style={"minWidth":"160px"}),
            html.Div([stl("Condition"),  rad("sd-cond","A",["A","B","C"])],                                    style={"minWidth":"90px"}),
            html.Div([stl("Camouflage"), rad("sd-cam","global",[("Global","global"),("Local","local"),("None","none")])],style={"minWidth":"140px"}),
            html.Div([stl("Metric"),     rad("sd-met","auc",[("AUC-ROC","auc"),("Avg Precision","ap")])],      style={"minWidth":"140px"}),
        ], style={"display":"flex","gap":"20px","flexWrap":"wrap"})],s={"marginBottom":"12px"})
        return html.Div([ctrl,
            html.Div([
                card([dcc.Graph(id="sd-box",   style={"height":"380px"})],s={"flex":"3"}),
                card([dcc.Graph(id="sd-lines", style={"height":"380px"})],s={"flex":"2"}),
            ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
            card([dcc.Graph(id="sd-std", style={"height":"240px"})]),
        ])

    # ── COLA ──────────────────────────────────────────────────────────────────
    elif tab == "cola":
        if cola.empty:
            return html.P("cola_graph_vs_noedge.csv not found. Run utils/test_cola.py first.",style={"color":MUTED})

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Global camouflage","Local camouflage"],
            horizontal_spacing=0.10)
        cross = {}
        for ci, cam in enumerate(["global","local"]):
            sub  = cola[cola.camouflage==cam].sort_values("gamma")
            c    = ACCENT if cam=="global" else PURPLE
            col_ = ci+1
            cx_rows = sub[sub["delta"]<=0]["gamma"]
            cx   = float(cx_rows.min()) if not cx_rows.empty else np.nan
            cross[cam] = cx

            fig.add_trace(go.Scatter(
                x=list(sub.gamma)+list(sub.gamma)[::-1],
                y=list(sub.auc_full)+list(sub.auc_bare)[::-1],
                fill="toself", fillcolor=c, opacity=0.07,
                line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=col_)
            for col_name, lbl, dash_, sym, sl in [
                ("auc_full","Full graph","solid","circle",ci==0),
                ("auc_bare","No edges (attr only)","dash","diamond",ci==0),
            ]:
                fig.add_trace(go.Scatter(
                    x=sub.gamma, y=sub[col_name], mode="lines+markers",
                    name=lbl, showlegend=sl,
                    line=dict(color=c,width=2.5,dash=dash_),
                    marker=dict(size=7,symbol=sym),
                    hovertemplate=f"<b>{lbl}</b><br>γ=%{{x:.1f}}<br>AUC=%{{y:.3f}}<extra></extra>"), row=1, col=col_)
            fig.add_trace(go.Scatter(
                x=sub.gamma, y=sub["delta"], mode="lines",
                name="Δ (full−bare)", showlegend=(ci==0),
                line=dict(color=AMBER,width=1.5,dash="dot"),
                hovertemplate="Δ=%{y:.3f}<extra></extra>"), row=1, col=col_)
            if not np.isnan(cx):
                fig.add_vline(x=cx, line_dash="dot", line_color=RED, line_width=1, row=1, col=col_)
            fig.add_hline(y=0.5, line_dash="dot", line_color=RED,   line_width=0.8, row=1, col=col_)
            fig.add_hline(y=0.0, line_dash="dot", line_color=MUTED, line_width=0.5, row=1, col=col_)

        fig.update_layout(**BASE, height=420,
            title="CoLA: full graph vs no-edges  (Δ>0 = graph helps · Δ<0 = graph hurts · red line = crossover)")
        fig.update_xaxes(gridcolor=BORDER, title="γ", tickfont=dict(color=MUTED))
        fig.update_yaxes(gridcolor=BORDER, range=[-0.65,1.05], tickfont=dict(color=MUTED))

        cross_str = " · ".join([f"{k}: γ={v:.1f}" for k,v in cross.items() if not np.isnan(v)])
        info = card([html.Div([
            html.Span("Crossover  ", style={"color":MUTED,"fontSize":"10px","letterSpacing":"2px"}),
            html.Span(cross_str,     style={"color":RED,"fontSize":"14px","fontWeight":"700"}),
            html.Span("  — beyond this γ, removing edges improves CoLA",
                      style={"color":MUTED,"fontSize":"10px"}),
        ])],s={"marginBottom":"12px","borderColor":RED+"44"})

        fd = html.Div()
        if not colafd.empty:
            fig_fd = go.Figure()
            for col_name, c, lbl in [("global_dist",ACCENT,"Global"),("local_dist",PURPLE,"Local")]:
                if col_name in colafd.columns:
                    fig_fd.add_trace(go.Scatter(
                        x=colafd.gamma, y=colafd[col_name], mode="lines+markers",
                        name=f"{lbl} attacker", line=dict(color=c,width=2), marker=dict(size=6),
                        hovertemplate=f"{lbl}<br>γ=%{{x:.1f}}<br>dist=%{{y:.3f}}<extra></extra>"))
            fig_fd.update_layout(**BASE, height=260,
                title="Feature distance anomaly→normal mean  (decreasing = camouflage working)",
                xaxis=AX("γ"), yaxis=AX("L2 distance"))
            fd = card([dcc.Graph(figure=fig_fd, style={"height":"260px"})],s={"marginTop":"12px"})

        return html.Div([info, card([dcc.Graph(figure=fig, style={"height":"420px"})]), fd])

    # ── STRUCTURE ─────────────────────────────────────────────────────────────
    elif tab == "structure":
        if rel.empty:
            return html.P("structure_reliance.csv not found. Run utils/test_structure_reliance.py first.",style={"color":MUTED})
        ctrl = card([html.Div([
            html.Div([stl("Camouflage"),          rad("str-cam","global",[("Global","global"),("Local","local")])],style={"minWidth":"140px"}),
            html.Div([stl("Detector (detail)"),   drp("str-det",RELDTS[0] if RELDTS else None,RELDTS)],            style={"minWidth":"200px"}),
        ], style={"display":"flex","gap":"24px"})],s={"marginBottom":"12px"})
        return html.Div([ctrl,
            card([dcc.Graph(id="str-rel",  style={"height":"360px"})],s={"marginBottom":"12px"}),
            html.Div([
                card([dcc.Graph(id="str-det1", style={"height":"320px"})],s={"flex":"1"}),
                card([dcc.Graph(id="str-det2", style={"height":"320px"})],s={"flex":"1"}),
            ], style={"display":"flex","gap":"12px"}),
        ])

    # ── REAL DATASETS ─────────────────────────────────────────────────────────
    elif tab == "real":
        if real.empty: return html.P("real_results.csv not found. Run run_real.py first.",style={"color":MUTED})
        ctrl = card([html.Div([
            html.Div([stl("Metric"), rad("rl-met","auc",[("AUC-ROC","auc"),("Avg Precision","ap")])],style={"minWidth":"140px"}),
            html.Div([stl("Sort"),   rad("rl-srt","mean",[("Mean AUC","mean"),("Fixed order","det")])],style={"minWidth":"180px"}),
        ], style={"display":"flex","gap":"24px"})],s={"marginBottom":"12px"})
        return html.Div([ctrl,
            html.Div([
                card([dcc.Graph(id="rl-bar",   style={"height":"400px"})],s={"flex":"3"}),
                card([dcc.Graph(id="rl-radar", style={"height":"400px"})],s={"flex":"2"}),
            ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
            card([dcc.Graph(id="rl-heat", style={"height":"340px"})]),
        ])

    # ── SBM vs REAL ───────────────────────────────────────────────────────────
    elif tab == "transfer":
        if sbm.empty or real.empty:
            return html.P("Need both sbm_results.csv and real_results.csv.",style={"color":MUTED})

        col  = acl(sbm)
        sbm0 = sbm[(sbm.gamma==0)&(sbm.condition=="A")&(sbm.camouflage=="global")].set_index("detector")
        rmu  = real.groupby("detector")["auc"].mean()
        rstd = real.groupby("detector")["auc"].std().fillna(0)
        sstd = sbm0["auc_std"] if "auc_std" in sbm0.columns else pd.Series(0,index=sbm0.index)
        common = [d for d in RDETS if d in sbm0.index and d not in LTD]

        rho = np.nan
        if len(common) > 2:
            rho, _ = stats.spearmanr([sbm0.loc[d,col] for d in common],[rmu[d] for d in common])

        # Scatter
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=[0.3,1],y=[0.3,1], mode="lines",
            line=dict(color=MUTED,dash="dash",width=1), showlegend=False, hoverinfo="skip"))
        for d in common:
            fig_sc.add_trace(go.Scatter(
                x=[sbm0.loc[d,col]], y=[rmu[d]], mode="markers+text",
                name=d, text=[d], textposition="top center",
                textfont=dict(color=dc(d),size=9),
                marker=dict(color=dc(d),size=13,line=dict(color=BG,width=2)),
                error_x=dict(type="data",array=[float(sstd.get(d,0))],color=MUTED,thickness=1.2,width=4),
                error_y=dict(type="data",array=[float(rstd.get(d,0))],color=MUTED,thickness=1.2,width=4),
                hovertemplate=f"<b>{d}</b><br>SBM@0={sbm0.loc[d,col]:.3f}<br>Real={rmu[d]:.3f}<extra></extra>",
                showlegend=False))
        fig_sc.update_layout(**BASE, height=420,
            title=f"SBM@γ=0 vs Real mean AUC  |  Spearman ρ={rho:.2f}  (diagonal = perfect transfer)",
            xaxis=AX("SBM AUC @ γ=0 · Cond A · global",r=[0.25,1.0]),
            yaxis=AX("Real-world mean AUC",             r=[0.25,1.0]))

        # Delta bar
        diffs = [(d, float(sbm0.loc[d,col])-float(rmu[d])) for d in common]
        diffs.sort(key=lambda x: x[1], reverse=True)
        det_d = [x[0] for x in diffs]
        val_d = [x[1] for x in diffs]
        fig_bar = go.Figure(go.Bar(
            x=det_d, y=val_d,
            marker_color=[RED if v>0 else GREEN for v in val_d],
            marker_line_width=0,
            text=[f"{v:+.3f}" for v in val_d],
            textposition="outside", textfont=dict(color=TEXT,size=9),
            hovertemplate="<b>%{x}</b><br>SBM−Real=%{y:+.3f}<extra></extra>"))
        fig_bar.add_hline(y=0, line_color=MUTED, line_width=0.8)
        fig_bar.update_layout(**BASE, height=320,
            title="SBM@γ=0 minus Real AUC  (red = SBM overestimates · green = SBM underestimates)",
            xaxis={**AX(), "tickangle":-30}, yaxis=AX("ΔAUC (SBM − Real)"))

        # Combined heatmap SBM@0 + all datasets
        x_cols = ["SBM@γ=0"] + RDSS
        z2, y2 = [], []
        for d in common:
            row_ = [float(sbm0.loc[d,col])]
            for ds in RDSS:
                m = real[(real.detector==d)&(real.dataset==ds)]["auc"]
                row_.append(float(m.mean()) if not m.empty else None)
            z2.append(row_)
            y2.append(d)
        ann2 = []
        for ri,row_ in enumerate(z2):
            for ci,v in enumerate(row_):
                if v is None: continue
                fc = "#111" if 0.3<=v<=0.7 else TEXT
                ann2.append(dict(x=x_cols[ci], y=y2[ri], text=f"{v:.2f}",
                    showarrow=False, font=dict(size=9,color=fc,family="monospace"),
                    xref="x",yref="y"))
        fig_hm = go.Figure(go.Heatmap(
            z=z2, x=x_cols, y=y2, colorscale="RdYlGn", zmin=0, zmax=1,
            colorbar=dict(title="AUC",tickfont=dict(color=TEXT,family="monospace"),
                          title_font=dict(color=TEXT,family="monospace")),
            hovertemplate="<b>%{y}</b> · %{x}<br>AUC=%{z:.3f}<extra></extra>"))
        fig_hm.update_layout(**BASE, height=360, annotations=ann2,
            title="AUC · SBM@γ=0 vs each real dataset  (same colorscale — compare columns directly)",
            xaxis=dict(tickfont=dict(color=TEXT,size=11)),
            yaxis=dict(tickfont=dict(color=TEXT,size=11)))

        rho_note = card([html.Div([
            html.Span("Spearman ρ = ",style={"color":MUTED,"fontSize":"11px"}),
            html.Span(f"{rho:.2f}",  style={"color":RED,"fontSize":"18px","fontWeight":"700"}),
            html.Span("  — near-zero: SBM ranking does not predict real-world ranking. "
                      "Detectors best on synthetic graphs (GCAD, GCAD+IDK, LOF) underperform on real data. "
                      "GAE and DONE transfer better.",
                      style={"color":MUTED,"fontSize":"11px"}),
        ])],s={"marginBottom":"12px","borderColor":RED+"44"})

        return html.Div([
            rho_note,
            html.Div([
                card([dcc.Graph(figure=fig_sc,  style={"height":"420px"})],s={"flex":"3"}),
                card([dcc.Graph(figure=fig_bar, style={"height":"420px"})],s={"flex":"2"}),
            ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
            card([dcc.Graph(figure=fig_hm, style={"height":"360px"})]),
        ])

    return html.P("Select a tab.",style={"color":MUTED})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("av-chart","figure"),
              Input("av-cond","value"), Input("av-cam","value"),
              Input("av-met","value"),  Input("av-bands","value"),
              Input("av-dets","value"))
def cb_av(cond, cam, met, bands, dets):
    if sbm.empty or not dets: return go.Figure()
    col     = acl(sbm) if met=="auc" else ("ap_mean" if "ap_mean" in sbm.columns else "ap")
    std_col = "auc_std" if met=="auc" else "ap_std"
    base_y  = 0.5 if met=="auc" else 0.05
    sub     = sbm[(sbm.condition==cond)&(sbm.camouflage==cam)&sbm.detector.isin(dets)]
    fig     = go.Figure()
    for d in [x for x in ORD if x in dets]:
        s = sub[sub.detector==d].sort_values("gamma")
        if s.empty: continue
        c = dc(d); dash_ = "dash" if d in LTD else "solid"; op = 0.45 if d in LTD else 1.0
        if bands=="show" and std_col in s.columns and d not in LTD:
            fig.add_trace(go.Scatter(
                x=list(s.gamma)+list(s.gamma)[::-1],
                y=list((s[col]+s[std_col]).clip(upper=1))+
                  list((s[col]-s[std_col]).clip(lower=0))[::-1],
                fill="toself", fillcolor=c, opacity=0.08,
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=s.gamma, y=s[col], mode="lines+markers", name=d,
            line=dict(color=c,width=2,dash=dash_), opacity=op,
            marker=dict(size=6,color=c,line=dict(color=BG,width=1)),
            hovertemplate=f"<b>{d}</b><br>γ=%{{x:.1f}}<br>{met.upper()}=%{{y:.3f}}<extra></extra>"))
    hln(fig, base_y)
    fig.update_layout(**BASE,
        title=f"{'AUC-ROC' if met=='auc' else 'Avg Precision'} vs γ · Cond {cond} · {cam}",
        xaxis=AX("γ",GAMMAS), yaxis=AX(met.upper(),r=[0,1.05]))
    return fig


@app.callback(Output("bmp-chart","figure"),
              Input("bmp-cond","value"), Input("bmp-cam","value"))
def cb_bump(cond, cam):
    if sbm.empty: return go.Figure()
    col  = acl(sbm)
    sub  = sbm[(sbm.condition==cond)&(sbm.camouflage==cam)]
    if sub.empty: return go.Figure()
    piv  = sub.pivot(index="detector", columns="gamma", values=col)
    rnks = piv.rank(ascending=False, axis=0)
    n    = len(piv)
    fig  = go.Figure()
    for d in [x for x in ORD if x in rnks.index]:
        g_    = sorted(rnks.columns)
        r_    = [rnks.loc[d,g] for g in g_]
        dash_ = "dash" if d in LTD else "solid"
        op    = 0.4  if d in LTD else 1.0
        fig.add_trace(go.Scatter(
            x=g_, y=r_, mode="lines+markers+text", name=d,
            line=dict(color=dc(d),width=2.5,dash=dash_), opacity=op,
            marker=dict(size=8,color=dc(d),line=dict(color=BG,width=1.5)),
            text=[None]*(len(g_)-1)+[d],
            textposition="middle right", textfont=dict(color=dc(d),size=10),
            hovertemplate=f"<b>{d}</b><br>γ=%{{x:.1f}}<br>rank=%{{y:.0f}}<extra></extra>"))
    bump_layout = {**BASE, "margin": dict(l=52,r=110,t=44,b=46)}
    fig.update_layout(**bump_layout,
        title=f"Detector ranking evolution · Cond {cond} · {cam}  (rank 1 = best AUC)",
        xaxis=AX("γ",GAMMAS),
        yaxis=dict(**AX("Rank"), autorange="reversed",
                   tickvals=list(range(1,n+1)), ticktext=[str(i) for i in range(1,n+1)]))
    return fig


@app.callback(Output("hm-chart","figure"),
              Input("hm-cond","value"), Input("hm-cam","value"),
              Input("hm-sort","value"), Input("hm-met","value"))
def cb_hm(cond, cam, sort_by, met):
    if sbm.empty: return go.Figure()
    col = acl(sbm) if met=="auc" else ("ap_mean" if "ap_mean" in sbm.columns else "ap")
    sub = sbm[(sbm.condition==cond)&(sbm.camouflage==cam)]
    if sub.empty: return go.Figure()
    go_ = {"none":0,"indirect":1,"direct":2}
    if sort_by=="auc0":
        order = sub[sub.gamma==0.0].sort_values(col,ascending=True)["detector"].tolist()
    elif sort_by=="mean":
        order = sub.groupby("detector")[col].mean().sort_values(ascending=True).index.tolist()
    else:
        order = sorted(sub["detector"].unique(),
                       key=lambda d:(go_.get(GRP.get(d,""),9), ORD.index(d) if d in ORD else 99))
    gammas = sorted(sub.gamma.unique())
    z = []
    for d in order:
        row_ = []
        for g in gammas:
            m = sub[(sub.detector==d)&(sub.gamma==g)]
            row_.append(float(m[col].values[0]) if len(m) else None)
        z.append(row_)
    ann = []
    for ri,row_ in enumerate(z):
        for ci,v in enumerate(row_):
            if v is None: continue
            fc = "#111" if 0.3<=v<=0.7 else TEXT
            ann.append(dict(x=f"γ={gammas[ci]:.1f}", y=order[ri], text=f"{v:.3f}",
                showarrow=False, font=dict(size=9,color=fc,family="monospace"), xref="x",yref="y"))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"γ={g:.1f}" for g in gammas], y=order,
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title=met.upper(),
                      tickfont=dict(color=TEXT,family="monospace"),
                      title_font=dict(color=TEXT,family="monospace")),
        hovertemplate="<b>%{y}</b><br>%{x}<br>val=%{z:.3f}<extra></extra>"))
    fig.update_layout(**BASE, height=500, annotations=ann,
        title=f"{'AUC' if met=='auc' else 'AP'} heatmap · Cond {cond} · {cam}",
        xaxis=dict(side="bottom",tickfont=dict(color=TEXT,size=11)),
        yaxis=dict(tickfont=dict(color=TEXT,size=11)))
    return fig


@app.callback(Output("sd-box","figure"), Output("sd-lines","figure"),
              Output("sd-std","figure"),
              Input("sd-det","value"), Input("sd-cond","value"),
              Input("sd-cam","value"),  Input("sd-met","value"))
def cb_seeds(det, cond, cam, met):
    empty = go.Figure()
    if raw.empty or not det: return empty, empty, empty
    sub = raw[(raw.detector==det)&(raw.condition==cond)&(raw.camouflage==cam)]
    if sub.empty: return empty, empty, empty
    col = met; c = dc(det)
    gammas_ = sorted(sub.gamma.unique())

    fig_box = go.Figure()
    for g in gammas_:
        vals = sub[sub.gamma==g][col].values
        fig_box.add_trace(go.Box(
            y=vals, name=f"{g:.1f}", marker_color=c,
            line=dict(color=c), fillcolor=hex_rgba(c, 0.2),
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(size=6,color=c,line=dict(color=BG,width=1)),
            showlegend=False,
            hovertemplate=f"γ={g:.1f}<br>{col.upper()}=%{{y:.3f}}<extra></extra>"))
    hln(fig_box, 0.5 if col=="auc" else 0.05)
    fig_box.update_layout(**BASE,
        title=f"{det} · {col.upper()} distribution per γ · Cond {cond} · {cam}",
        xaxis=AX("γ"), yaxis=AX(col.upper(),r=[0,1.05]))

    fig_ln = go.Figure()
    for seed in sorted(sub.seed.unique()):
        ss = sub[sub.seed==seed].sort_values("gamma")
        fig_ln.add_trace(go.Scatter(
            x=ss.gamma, y=ss[col], mode="lines+markers", name=f"seed {seed}",
            line=dict(color=c,width=1.2), opacity=0.4, marker=dict(size=4),
            hovertemplate=f"seed {seed}<br>γ=%{{x:.1f}}<br>{col.upper()}=%{{y:.3f}}<extra></extra>"))
    mu = sub.groupby("gamma")[col].mean().reset_index()
    sd = sub.groupby("gamma")[col].std().reset_index()
    fig_ln.add_trace(go.Scatter(
        x=list(mu.gamma)+list(mu.gamma)[::-1],
        y=list((mu[col]+sd[col]).clip(upper=1))+
          list((mu[col]-sd[col]).clip(lower=0))[::-1],
        fill="toself", fillcolor=c, opacity=0.12,
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_ln.add_trace(go.Scatter(
        x=mu.gamma, y=mu[col], mode="lines+markers", name="mean",
        line=dict(color=c,width=3), marker=dict(size=9,line=dict(color=BG,width=2)),
        hovertemplate=f"<b>mean</b><br>γ=%{{x:.1f}}<br>{col.upper()}=%{{y:.3f}}<extra></extra>"))
    hln(fig_ln, 0.5 if col=="auc" else 0.05)
    fig_ln.update_layout(**BASE,
        title=f"{det} · individual seeds + mean ± std band",
        xaxis=AX("γ",gammas_), yaxis=AX(col.upper(),r=[0,1.05]))

    fig_std = go.Figure(go.Scatter(
        x=sd.gamma, y=sd[col], mode="lines+markers+text",
        line=dict(color=AMBER,width=2), marker=dict(size=7,color=AMBER),
        text=[f"{v:.3f}" for v in sd[col]], textposition="top center",
        textfont=dict(color=AMBER,size=9), showlegend=False,
        hovertemplate="γ=%{x:.1f}<br>std=%{y:.3f}<extra></extra>"))
    fig_std.update_layout(**BASE, height=240,
        title=f"{det} · seed std per γ  (high = detector is unstable at this camouflage level)",
        xaxis=AX("γ",gammas_), yaxis=AX("std"))
    return fig_box, fig_ln, fig_std


@app.callback(Output("str-rel","figure"), Input("str-cam","value"))
def cb_str_rel(cam):
    if rel.empty: return go.Figure()
    sub = rel[rel.camouflage==cam]
    det_clr = {"DOMINANT":ACCENT,"AnomalyDAE":PURPLE,"DONE":"#f472b6","CoLA":CYAN}
    fig = go.Figure()
    for d in RELDTS:
        s = sub[sub.detector==d].sort_values("gamma")
        c = det_clr.get(d,"#888")
        if "reliance_std" in s.columns:
            fig.add_trace(go.Scatter(
                x=list(s.gamma)+list(s.gamma)[::-1],
                y=list(s.reliance+s.reliance_std)+
                  list((s.reliance-s.reliance_std).clip(lower=-1))[::-1],
                fill="toself", fillcolor=c, opacity=0.08,
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=s.gamma, y=s.reliance, mode="lines+markers", name=d,
            line=dict(color=c,width=2.5), marker=dict(size=7),
            hovertemplate=f"<b>{d}</b><br>γ=%{{x:.1f}}<br>reliance=%{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    fig.update_layout(**BASE,
        title=f"Structure reliance = full AUC − no-edges AUC  [{cam}]  (>0 graph helps · <0 graph hurts)",
        xaxis=AX("γ",GAMMAS), yaxis=AX("Reliance"))
    return fig


@app.callback(Output("str-det1","figure"), Output("str-det2","figure"),
              Input("str-det","value"))
def cb_str_det(det):
    empty = go.Figure()
    if rel.empty or not det: return empty, empty
    styles = {"full":(ACCENT,"solid"),"no_edges":(PURPLE,"dash"),"rand_feats":(AMBER,"dot")}
    labels = {"full":"Full graph","no_edges":"No edges (attr only)","rand_feats":"Rand feats (struct only)"}
    figs = []
    for cam in ["global","local"]:
        s = rel[(rel.detector==det)&(rel.camouflage==cam)].sort_values("gamma")
        fig = go.Figure()
        for ck,(c,dash_) in styles.items():
            if ck not in s.columns: continue
            sc_ = f"{ck}_std"
            if sc_ in s.columns:
                fig.add_trace(go.Scatter(
                    x=list(s.gamma)+list(s.gamma)[::-1],
                    y=list((s[ck]+s[sc_]).clip(upper=1))+
                      list((s[ck]-s[sc_]).clip(lower=0))[::-1],
                    fill="toself", fillcolor=c, opacity=0.10,
                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=s.gamma, y=s[ck], mode="lines+markers", name=labels[ck],
                line=dict(color=c,width=2,dash=dash_), marker=dict(size=6),
                hovertemplate=f"<b>{labels[ck]}</b><br>γ=%{{x:.1f}}<br>AUC=%{{y:.3f}}<extra></extra>"))
        hln(fig, 0.5)
        fig.update_layout(**BASE,
            title=f"{det} · {cam} camouflage  (solid=full · dash=no-edges · dot=rand-feats)",
            xaxis=AX("γ",GAMMAS), yaxis=AX("AUC",r=[0,1.05]))
        figs.append(fig)
    return figs[0], figs[1]


@app.callback(Output("rl-bar","figure"), Output("rl-radar","figure"),
              Output("rl-heat","figure"),
              Input("rl-met","value"), Input("rl-srt","value"))
def cb_real(met, srt):
    empty = go.Figure()
    if real.empty: return empty, empty, empty
    col = met; std_col = f"{col}_std"
    if srt=="mean":
        mu    = real.groupby("detector")[col].mean()
        order = [d for d in mu.sort_values(ascending=False).index if d in RDETS]
    else:
        order = RDETS

    fig_bar = go.Figure()
    for ds in RDSS:
        sub  = real[real.dataset==ds].set_index("detector").reindex(order).dropna()
        errs = sub[std_col].values if std_col in sub.columns else [0]*len(sub)
        fig_bar.add_trace(go.Bar(
            x=sub.index, y=sub[col], name=ds, marker_color=DSC.get(ds,"#888"),
            error_y=dict(type="data",array=list(errs),color=MUTED,thickness=1.2,width=3),
            hovertemplate=f"<b>%{{x}}</b> · {ds}<br>{col.upper()}=%{{y:.3f}}<extra></extra>"))
    hln(fig_bar, 0.5 if met=="auc" else 0.05)
    fig_bar.update_layout(**BASE, barmode="group",
        title=f"{'AUC-ROC' if col=='auc' else 'Avg Precision'} · real datasets per detector",
        xaxis={**AX(), "tickangle":-35}, yaxis=AX(col.upper(),r=[0,1.05]))

    fig_rad = go.Figure()
    cats = RDSS + [RDSS[0]]
    for d in order[:9]:
        vals = [real[(real.detector==d)&(real.dataset==ds)][col].mean() for ds in RDSS]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats, mode="lines+markers", name=d,
            line=dict(color=dc(d),width=2), fill="toself", fillcolor=dc(d), opacity=0.07,
            marker=dict(size=5),
            hovertemplate=f"<b>{d}</b><br>%{{theta}}<br>{col.upper()}=%{{r:.3f}}<extra></extra>"))
    fig_rad.update_layout(**BASE, height=400,
        title="Detector profile across real datasets",
        polar=dict(bgcolor=CARD,
                   radialaxis=dict(range=[0,1],gridcolor=BORDER,tickfont=dict(color=MUTED,size=9)),
                   angularaxis=dict(gridcolor=BORDER,tickfont=dict(color=TEXT,size=11))))

    z = []
    for d in order:
        row_ = [real[(real.detector==d)&(real.dataset==ds)][col].mean()
                if not real[(real.detector==d)&(real.dataset==ds)].empty else None
                for ds in RDSS]
        z.append(row_)
    ann = []
    for ri,row_ in enumerate(z):
        for ci,v in enumerate(row_):
            if v is None: continue
            fc = "#111" if 0.3<=v<=0.7 else TEXT
            r_ = real[(real.detector==order[ri])&(real.dataset==RDSS[ci])]
            zs = (std_col in r_.columns and not r_.empty and r_[std_col].values[0]==0)
            ann.append(dict(x=RDSS[ci], y=order[ri],
                text=f"{v:.2f}{'*' if zs else ''}",
                showarrow=False, font=dict(size=9,color=fc,family="monospace"),
                xref="x",yref="y"))
    fig_hm = go.Figure(go.Heatmap(
        z=z, x=RDSS, y=order, colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title=col.upper(),
                      tickfont=dict(color=TEXT,family="monospace"),
                      title_font=dict(color=TEXT,family="monospace")),
        hovertemplate="<b>%{y}</b> · %{x}<br>val=%{z:.3f}<extra></extra>"))
    fig_hm.update_layout(**BASE, height=340, annotations=ann,
        title=f"{'AUC' if met=='auc' else 'AP'} · detector × dataset  (*=std=0, deterministic)",
        xaxis=dict(tickfont=dict(color=TEXT,size=12)),
        yaxis=dict(tickfont=dict(color=TEXT,size=12)))
    return fig_bar, fig_rad, fig_hm


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Project root : {PROJECT_ROOT}")
    for f in ["sbm_results.csv","sbm_raw.csv","structure_reliance.csv",
              "cola_graph_vs_noedge.csv","real_results.csv"]:
        p = RESULTS/f
        print(f"  {'✓' if p.exists() else '✗'} {f}")
    print("\n  → http://127.0.0.1:8050\n")
    app.run(debug=False, use_reloader=False)