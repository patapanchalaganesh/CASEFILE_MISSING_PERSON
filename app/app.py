import os, sys, streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import streamlit.components.v1 as components, joblib, folium
from streamlit_folium import st_folium
from folium import plugins

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT); sys.path.insert(0, PROJECT_ROOT)

# ═══════════════ CONFIG ═══════════════
st.set_page_config(page_title='CASEFILE | Pan-India Missing Person Investigation',
                   page_icon='📍', layout='wide', initial_sidebar_state='expanded')

# ═══════════════ CSS + ANIMATIONS ═══════════════
st.markdown("""<style>
@keyframes fadeInUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes slideInLeft{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
@keyframes pulseRed{0%,100%{opacity:1}50%{opacity:0.6}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}

.stApp{background:#f8f9fc;font-family:'Segoe UI',system-ui,-apple-system,sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#ffffff,#f0f4f9);border-right:1px solid #e2e8f0}
section[data-testid="stSidebar"]>div{padding-top:0}
#MainMenu,footer,header{visibility:hidden}

.kpi{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:0.9rem 1.1rem;
     margin-bottom:0.5rem;border-top:3px solid #3b82f6;position:relative;overflow:hidden;
     animation:fadeInUp 0.5s ease-out both;transition:all 0.3s;
     box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.kpi::after{content:'';position:absolute;top:0;left:-100%;width:200%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(59,130,246,0.04),transparent);
    animation:shimmer 4s ease-in-out infinite}
.kpi:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(59,130,246,0.12);border-color:#3b82f6}
.kpi-l{font-size:0.62rem;color:#6b7c93;text-transform:uppercase;letter-spacing:1.5px;margin:0}
.kpi-v{font-size:1.4rem;font-weight:700;color:#1e293b;margin:0.2rem 0 0}

.banner{background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #ef4444;
        border-radius:12px;padding:1.3rem 2rem;margin-bottom:1.2rem;
        animation:fadeInUp 0.4s ease-out both;box-shadow:0 2px 8px rgba(0,0,0,0.05)}
.badge-active{background:#fef2f2;color:#dc2626;padding:4px 14px;border-radius:20px;
    font-size:0.65rem;font-weight:700;border:1px solid #fca5a5;letter-spacing:1px;
    animation:pulseRed 2s ease-in-out infinite}

.panel{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.3rem;
    color:#475569;font-size:0.87rem;line-height:1.7;animation:fadeIn 0.5s ease-out both;
    transition:all 0.3s;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.panel:hover{border-color:#93c5fd;box-shadow:0 4px 12px rgba(59,130,246,0.08)}

.stitle{color:#1e293b;font-size:1.05rem;font-weight:700;margin:1.2rem 0 0.4rem;padding-bottom:5px;
    border-bottom:2px solid #3b82f6;display:inline-block;animation:slideInLeft 0.4s ease-out both}

.rank-c{background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid;
    border-radius:12px;padding:0.9rem 1.3rem;margin:0.5rem 0;
    animation:fadeInUp 0.5s ease-out both;transition:all 0.3s;
    box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.rank-c:hover{transform:translateX(4px);box-shadow:0 4px 15px rgba(0,0,0,0.08)}

.disclaimer{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
    padding:0.6rem 1rem;font-size:0.7rem;color:#92400e;margin-top:0.8rem}
.sidebar-info{color:#64748b;font-size:0.75rem;margin:3px 0}
.sidebar-val{color:#2563eb;font-weight:600}
.sidebar-logo{animation:float 3s ease-in-out infinite;display:inline-block}

.dossier-table{width:100%;border-collapse:separate;border-spacing:0;font-size:0.82rem;
    border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.dossier-table tr{transition:background 0.2s}
.dossier-table tr:hover{background:#f1f5f9}
.dossier-table td{padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#334155}
.dossier-table td:first-child{color:#64748b;font-weight:600;width:42%;background:#f8fafc}
.dossier-table td:last-child{color:#1e293b;font-weight:500}

.js-plotly-plot .plotly .main-svg{background:transparent!important}
.stTabs [data-baseweb="tab-list"]{gap:5px;padding:4px;background:#f1f5f9;border-radius:10px}
.stTabs [data-baseweb="tab"]{background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
    color:#64748b;padding:10px 18px;font-weight:600;font-size:0.78rem;transition:all 0.3s}
.stTabs [data-baseweb="tab"]:hover{color:#3b82f6;background:#eff6ff;border-color:#93c5fd}
.stTabs [aria-selected="true"]{background:#eff6ff;color:#2563eb;border-color:#3b82f6;
    box-shadow:0 2px 8px rgba(59,130,246,0.15);font-weight:700}
</style>""", unsafe_allow_html=True)

# ═══════════════ HELPERS ═══════════════
PL = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#64748b',
          title_font_color='#1e293b',xaxis=dict(gridcolor='#f1f5f9'),yaxis=dict(gridcolor='#f1f5f9'),
          legend=dict(font=dict(color='#64748b')),margin=dict(l=40,r=20,t=50,b=40))
ESRI='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'

def lcsv(p): return pd.read_csv(p) if os.path.exists(p) else None
def lcsv_s(p,n=15000): return pd.read_csv(p,nrows=n) if os.path.exists(p) else None
def lpkl(p): return joblib.load(p) if os.path.exists(p) else None

def K(l,v,icon="",clr="#3b82f6"):
    st.markdown(f'<div class="kpi" style="border-top-color:{clr}"><p class="kpi-l">{icon} {l}</p><p class="kpi-v">{v}</p></div>',unsafe_allow_html=True)

def S(t,sub=""):
    s=f' <span style="color:#94a3b8;font-size:0.72rem;">({sub})</span>' if sub else ""
    st.markdown(f'<p class="stitle">{t}{s}</p>',unsafe_allow_html=True)

def P(fig): fig.update_layout(**PL); st.plotly_chart(fig,use_container_width=True)

def html_map(p,h=550):
    if os.path.exists(p):
        with open(p,'r',encoding='utf-8') as f: components.html(f.read(),height=h,scrolling=True)
    else: st.warning(f"Map not found: {p}")

def gauge_chart(value, title="Search Priority Score"):
    fig = go.Figure(go.Indicator(mode="gauge+number",value=value,
        number=dict(font=dict(size=42,color='#1e293b')),
        title=dict(text=title,font=dict(size=13,color='#64748b')),
        gauge=dict(axis=dict(range=[0,100],tickwidth=1,tickcolor='#cbd5e1',dtick=20,
                   tickfont=dict(color='#94a3b8',size=10)),
            bar=dict(color='#3b82f6',thickness=0.3), bgcolor='#f1f5f9', borderwidth=0,
            steps=[dict(range=[0,30],color='#dcfce7'),dict(range=[30,60],color='#fef9c3'),
                   dict(range=[60,80],color='#ffedd5'),dict(range=[80,100],color='#fee2e2')],
            threshold=dict(line=dict(color='#ef4444',width=3),thickness=0.8,value=value))))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',height=240,margin=dict(l=30,r=30,t=50,b=20))
    return fig

# ═══════════════ LOAD DATA ═══════════════
cases_df = lcsv("data/synthetic/cases.csv")
inv_case = lcsv("data/synthetic/investigation_case.csv")
areas = lcsv("data/processed/area_centers.csv")
priority = lcsv("data/processed/search_priority.csv")
model_comp = lcsv("reports/model_comparison.csv")
routes = lcsv("data/processed/predicted_routes.csv")
traj = lcsv("data/processed/trajectory_summary.csv")
stay = lcsv("data/processed/stay_points_clustered.csv")
profiles = lcsv("data/processed/user_profiles.csv")

# Build full case list
all_cases = pd.DataFrame()
if inv_case is not None:
    all_cases = pd.concat([inv_case, cases_df], ignore_index=True) if cases_df is not None else inv_case.copy()
elif cases_df is not None:
    all_cases = cases_df.copy()

# ═══════════════ SIDEBAR ═══════════════
st.sidebar.markdown("""<div style="text-align:center;padding:1rem 0 0.4rem">
    <div class="sidebar-logo" style="font-size:2rem">📍</div>
    <div style="font-size:1.1rem;font-weight:800;color:#1e293b;letter-spacing:3px">Indian Case Dossier</div>
    <div style="font-size:0.58rem;color:#3b82f6;letter-spacing:2px;margin-top:2px">AI INVESTIGATION SYSTEM</div>
</div>""",unsafe_allow_html=True)
st.sidebar.markdown("---")

# State Jurisdiction Selector
states_list = ["All 28 States"]
if not all_cases.empty and 'State' in all_cases.columns:
    states_list += sorted(all_cases['State'].unique().tolist())
selected_state = st.sidebar.selectbox("Select State Jurisdiction", states_list)

# Filter cases by state
filtered_cases = all_cases.copy()
if selected_state != "All 28 States" and 'State' in all_cases.columns:
    filtered_cases = all_cases[all_cases['State'] == selected_state]

# Case ID selector
case_ids = filtered_cases['Case_ID'].tolist() if not filtered_cases.empty and 'Case_ID' in filtered_cases.columns else ["MP-AP-2026-001"]
selected_case = st.sidebar.selectbox("Select Case ID / FIR", case_ids)

# Get selected case
case = None
if not filtered_cases.empty and 'Case_ID' in filtered_cases.columns:
    match = filtered_cases[filtered_cases['Case_ID'] == selected_case]
    if not match.empty:
        case = match.iloc[0]

# Sidebar case info
if case is not None:
    st.sidebar.markdown("---")
    fir = case.get('FIR_Reference', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">FIR Reference: <span class="sidebar-val">{fir}</span></p>', unsafe_allow_html=True)
    ps = case.get('Police_Station', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">Police Station: <span class="sidebar-val">{ps}</span></p>', unsafe_allow_html=True)
    state = case.get('State', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">State / UT: <span class="sidebar-val">{state}</span></p>', unsafe_allow_html=True)
    pid = case.get('Person_ID', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">Missing Person: <span class="sidebar-val">IND-USR-{pid}</span></p>', unsafe_allow_html=True)
    ag = case.get('Age_Group', 'N/A')
    gen = case.get('Gender', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">Age Group: <span class="sidebar-val">{ag}</span> | Gender: <span class="sidebar-val">{gen}</span></p>', unsafe_allow_html=True)
    lst = case.get('Last_Seen_Time', 'N/A')
    day = case.get('Day', 'N/A')
    st.sidebar.markdown(f'<p class="sidebar-info">Last Seen: <span class="sidebar-val">{lst} IST</span> ({day})</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="disclaimer">CASEFILE Pan-India Missing Person System -- Academic Simulation covering all 28 States and 8 Union Territories of India.</div>', unsafe_allow_html=True)

# ═══════════════ MAIN TITLE ═══════════════
st.markdown("""<div style="margin-bottom:0.5rem">
    <span style="font-size:1.6rem;font-weight:800;color:#1e293b;letter-spacing:1px">
    📍 CASEFILE: Indian Missing Person Investigation System</span><br>
    <span style="color:#64748b;font-size:0.85rem">AI-Powered Geographical Location & Route Prediction System (28 States of India)</span>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer" style="margin-bottom:1rem">Academic & Decision-Support Simulation Only: Predictions are probabilistic estimates based on simulated historical trajectories across Indian States and must not be used for real-world police operations without human investigation.</div>', unsafe_allow_html=True)

# Ensure case is always set
if case is None and not all_cases.empty:
    case = all_cases.iloc[0]

# ═══════════════ TABS ═══════════════
tabs = st.tabs([
    "📋 FIR & Case Profile",
    "🎯 Probable Locations (ML)",
    "🛤️ Route Prediction (Markov)",
    "⚠️ Anomaly Detection",
    "🔍 Search Priority Matrix",
    "🗺️ Interactive Tactical Map",
    "💡 Explainability (XAI)",
    "📈 Model Evaluation",
])

# ═══════════════ TAB 1: FIR & CASE PROFILE ═══════════════
with tabs[0]:
    # Banner
    st.markdown(f"""<div class="banner">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="font-size:1.5rem;font-weight:800;color:#1e293b;letter-spacing:2px">
                    CASE {case.get('Case_ID','MP-AP-2026-001')}</span>
                <span class="badge-active" style="margin-left:12px">&#9679; ACTIVE</span>
                <p style="color:#64748b;font-size:0.75rem;margin:0.3rem 0 0">
                    MISSING PERSON &mdash; AI-POWERED INVESTIGATION SYSTEM</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    S("📋 Official FIR Case Profile & Investigation Overview")

    total_cases = len(all_cases) if not all_cases.empty else 112
    time_elapsed = case.get('Time_Since_Last_Seen', 48)
    pri_score = priority['composite_score'].max() if priority is not None and not priority.empty else 75
    pri_tier = str(priority.iloc[0].get('priority','High')).upper() if priority is not None and not priority.empty else 'HIGH'

    k1,k2,k3,k4 = st.columns(4)
    with k1: K("Registered Pan-India Cases", str(total_cases), "📊", "#2563eb")
    with k2: K("Time Elapsed", f"{time_elapsed} Hours", "~", "#f59e0b")
    with k3: K("Search Priority Score", f"{pri_score:.1f} / 100", "~", "#ef4444")
    with k4: K("Search Priority Tier", pri_tier, "~", "#dc2626")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])

    with left:
        S("📝 Case Dossier Specifications")
        lat = float(case.get('Last_Latitude', 17.39))
        lon = float(case.get('Last_Longitude', 78.49))
        dossier = {
            "Case ID": case.get('Case_ID','--'),
            "FIR Reference": case.get('FIR_Reference','--'),
            "Police Station Jurisdiction": case.get('Police_Station','--'),
            "State / UT": case.get('State','--'),
            "Anonymous Subject ID": f"IND-USR-{case.get('Person_ID','--')}",
            "Demographic Age Group": case.get('Age_Group','--'),
            "Gender": case.get('Gender','--'),
            "Last Known Signal Coordinates": f"{lat:.5f} N, {lon:.5f} E",
            "Time of Disappearance": f"{case.get('Last_Seen_Time','--')} IST",
            "Day of Week": case.get('Day','--'),
            "Weather Condition at Disappearance": case.get('Weather','--'),
            "Usual Frequent Area": str(case.get('Usual_Area','--')),
            "Previous Visited Area": str(case.get('Previous_Area','--')),
            "Simulated Target Destination": str(case.get('Target_Area','--')),
        }
        rows = "".join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in dossier.items()])
        st.markdown(f'<table class="dossier-table">{rows}</table>', unsafe_allow_html=True)

    with right:
        S("📊 Movement Metrics")
        m1, m2 = st.columns(2)
        with m1: K("Avg Historical Travel Distance", f"{case.get('Average_Distance',0):.2f} km", "~", "#22c55e")
        with m2: K("Avg Historical Movement Speed", f"{case.get('Average_Speed',0):.2f} km/h", "~", "#8b5cf6")

        S("Search Priority Gauge")
        fig = gauge_chart(pri_score)
        st.plotly_chart(fig, use_container_width=True)

    # Case Description (if available)
    desc = case.get('description', None)
    if desc and pd.notna(desc):
        S("📝 Case Narrative")
        st.markdown(f'<div class="panel">{desc}</div>', unsafe_allow_html=True)

    # Last Known Location Map
    st.markdown("<br>", unsafe_allow_html=True)
    S("🗺️ Last Known Signal Location")
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles=ESRI, attr='Esri')
    folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='exclamation-sign'),
                  popup=f"Last Known: {lat:.4f} N, {lon:.4f} E").add_to(m)
    folium.Circle([lat, lon], radius=1000, color='#ef4444', fill=True,
                  fill_opacity=0.1, popup="1km Search Radius").add_to(m)
    st_folium(m, height=380, returned_objects=[])
    st.caption(f"Coordinates: {lat:.5f} N, {lon:.5f} E")


# ═══════════════ TAB 2: PROBABLE LOCATIONS ═══════════════
with tabs[1]:
    S("🎯 Probable Location Prediction", "Multi-model ML classification")

    if model_comp is not None:
        best = model_comp.loc[model_comp['F1'].idxmax()]
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: K("Best Model", str(best['Model']), "~", "#22c55e")
        with k2: K("Accuracy", f"{best['Accuracy']:.0%}", "~", "#2563eb")
        with k3: K("F1 Score", f"{best['F1']:.3f}", "~", "#f59e0b")
        with k4: K("Top-3 Acc", f"{best['Top-3 Acc']:.0%}", "~", "#8b5cf6")
        with k5: K("Top-5 Acc", f"{best['Top-5 Acc']:.0%}", "~", "#ef4444")

        S("📊 Model Performance Comparison")
        colors = ['#2563eb','#22c55e','#f59e0b','#8b5cf6']
        fig = go.Figure()
        for i, (_, r) in enumerate(model_comp.iterrows()):
            fig.add_trace(go.Bar(name=r['Model'],
                x=['Accuracy','Precision','Recall','F1','Top-3','Top-5'],
                y=[r['Accuracy'],r['Precision'],r['Recall'],r['F1'],r['Top-3 Acc'],r['Top-5 Acc']],
                marker_color=colors[i%4]))
        fig.update_layout(barmode='group',title="4-Model Benchmark",yaxis=dict(range=[0,1.05]))
        P(fig)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("reports/feature_importance_GradientBoosting.png"):
            S("📈 Feature Importance"); st.image("reports/feature_importance_GradientBoosting.png")
    with c2:
        if os.path.exists("reports/confusion_matrix_GradientBoosting.png"):
            S("Confusion Matrix"); st.image("reports/confusion_matrix_GradientBoosting.png")

    if cases_df is not None and 'Target_Area' in cases_df.columns:
        S("📍 Case Distribution by Target Region")
        cnt = cases_df['Target_Area'].value_counts().reset_index()
        cnt.columns = ['Region','Cases']
        fig = px.pie(cnt, names='Region', values='Cases', hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2, title="Pan-India Case Distribution")
        P(fig)

    if cases_df is not None and 'State' in cases_df.columns:
        S("📍 Cases by State")
        sc = cases_df['State'].value_counts().reset_index()
        sc.columns = ['State','Cases']
        fig = px.bar(sc, x='State', y='Cases', color='Cases', color_continuous_scale='Blues',
                     title="Cases Registered per State")
        fig.update_layout(xaxis=dict(tickangle=-45))
        P(fig)


# ═══════════════ TAB 3: ROUTE PREDICTION ═══════════════
with tabs[2]:
    S("🛤️ Route Prediction -- Markov Chain", "Transition model with beam search decoding")

    tm = lpkl("models/transition_matrix.pkl")
    if tm is not None and hasattr(tm,'shape'):
        n = tm.shape[0]
        labels = [f"Region {i}" for i in range(n)]
        if areas is not None:
            labels = areas['name'].tolist()[:n]
        fig = px.imshow(tm, x=labels, y=labels, color_continuous_scale='Blues',
                        title="Regional Transition Probability Matrix",
                        labels=dict(x="To Region",y="From Region",color="P"))
        fig.update_layout(height=420)
        P(fig)

    if routes is not None:
        S("Top Predicted Routes")
        for rank in sorted(routes['route_rank'].unique())[:3]:
            route = routes[routes['route_rank']==rank].sort_values('step')
            prob = route['cumulative_probability'].iloc[-1]
            names = []
            for _, r in route.iterrows():
                if areas is not None:
                    match = areas[areas['area_id']==r['area_id']]
                    nm = match['name'].values[0] if len(match) else f"Region {r['area_id']}"
                else: nm = str(r.get('area_name', f"Region {r['area_id']}"))
                names.append(nm)
            path = " -> ".join(names)
            clr = ['#22c55e','#2563eb','#8b5cf6'][rank-1]
            st.markdown(f"""<div class="rank-c" style="border-color:{clr}">
                <span style="font-size:1.2rem;font-weight:800;color:{clr}">#{rank}</span>
                <span style="color:#1e293b;font-size:0.85rem;margin-left:8px">{path}</span>
                <span style="float:right;color:{clr};font-weight:700">{prob:.4f}</span>
            </div>""",unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("reports/transition_matrix.png"):
            st.image("reports/transition_matrix.png",caption="Transition Heatmap")
    with c2:
        if os.path.exists("reports/transition_graph.png"):
            st.image("reports/transition_graph.png",caption="Transition Network")


# ═══════════════ TAB 4: ANOMALY DETECTION ═══════════════
with tabs[3]:
    S("Anomaly Detection", "Ensemble: Isolation Forest + LOF + One-Class SVM")
    st.markdown('<div class="disclaimer" style="color:#dc2626;border-color:#fca5a5;background:#fef2f2">Statistical anomalies indicate unusual movement patterns only. NOT suspicious behavior.</div>',unsafe_allow_html=True)

    anom = lcsv_s("data/processed/gps_anomalies.csv", 20000)
    if anom is not None:
        col = 'ensemble_anomaly' if 'ensemble_anomaly' in anom.columns else 'is_anomaly'
        total = len(anom); n_a = int(anom[col].sum()) if col in anom.columns else 0
        k1,k2,k3,k4 = st.columns(4)
        with k1: K("Sample Points", f"{total:,}", "~", "#2563eb")
        with k2: K("Anomalies", f"{n_a:,}", "~", "#ef4444")
        with k3: K("Anomaly Rate", f"{(n_a/max(total,1))*100:.1f}%", "~", "#f59e0b")
        with k4: K("Ensemble", "2-of-3 Vote", "~", "#8b5cf6")

        c1, c2 = st.columns(2)
        with c1:
            if 'speed_kmh' in anom.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=anom[anom[col]==False]['speed_kmh'].clip(0,80),name='Normal',marker_color='#2563eb',opacity=0.7,nbinsx=40))
                fig.add_trace(go.Histogram(x=anom[anom[col]==True]['speed_kmh'].clip(0,80),name='Anomalous',marker_color='#ef4444',opacity=0.7,nbinsx=40))
                fig.update_layout(barmode='overlay',title="Speed: Normal vs Anomalous"); P(fig)
        with c2:
            if 'latitude' in anom.columns:
                s = anom.sample(min(2000,len(anom)))
                fig = px.scatter(s,x='longitude',y='latitude',color=col,opacity=0.5,
                    color_discrete_map={True:'#ef4444',False:'#2563eb'},title="Spatial Distribution"); P(fig)
    if os.path.exists("reports/anomaly_distributions.png"):
        S("Score Distributions"); st.image("reports/anomaly_distributions.png")


# ═══════════════ TAB 5: SEARCH PRIORITY ═══════════════
with tabs[4]:
    S("Search Priority Matrix", "6-factor composite scoring")
    if priority is not None:
        pri2 = priority.copy()
        if areas is not None and 'area_id' in pri2.columns:
            pri2 = pri2.merge(areas[['area_id','name']],on='area_id',how='left')
            if 'name' in pri2.columns:
                pri2['area_name']=pri2['name'].fillna(pri2['area_name']); pri2.drop(columns=['name'],inplace=True)

        for i, row in pri2.iterrows():
            sc=row['composite_score']; lv=str(row.get('priority','Medium'))
            clr='#ef4444' if 'Very' in lv else '#f59e0b' if 'High' in lv else '#2563eb'
            ar=row.get('area_name',f"Region {row['area_id']}")
            st.markdown(f"""<div class="rank-c" style="border-color:{clr}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div><span style="font-size:1.4rem;font-weight:800;color:{clr}">#{i+1}</span>
                        <span style="font-size:0.95rem;color:#1e293b;font-weight:600;margin-left:10px">{ar}</span>
                        <span style="background:{clr}20;color:{clr};padding:2px 10px;border-radius:12px;
                              font-size:0.62rem;font-weight:700;border:1px solid {clr}50;margin-left:8px">{lv.upper()}</span></div>
                    <div><span style="font-size:1.7rem;font-weight:800;color:{clr}">{sc:.1f}</span>
                        <span style="color:#94a3b8;font-size:0.7rem">/100</span></div>
                </div>
            </div>""",unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            S("Factor Radar -- Top Region")
            top = pri2.iloc[0]
            cats=['ML Score','Visit Freq','Route','Distance','Time Match','Anomaly']
            vals=[top.get(k,0) for k in ['ml_score','visit_score','route_score','distance_score','time_score','anomaly_score']]
            fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill='toself',
                fillcolor='rgba(37,99,235,0.2)',line=dict(color='#2563eb',width=2)))
            fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True,range=[0,100],gridcolor='#152040',tickfont=dict(color='#5a7099')),
                angularaxis=dict(gridcolor='#152040',tickfont=dict(color='#b0bed3',size=11))),
                showlegend=False,height=360); P(fig)
        with c2:
            S("Priority Gauge")
            fig=gauge_chart(float(pri2.iloc[0]['composite_score']),pri2.iloc[0].get('area_name','Top'))
            st.plotly_chart(fig,use_container_width=True)


# ═══════════════ TAB 6: TACTICAL MAP ═══════════════
with tabs[5]:
    S("Interactive Tactical Map", "Multi-layer Folium with evidence overlays")
    mtabs = st.tabs(["Investigation Map", "Movement Heatmap", "Cluster Map"])
    with mtabs[0]:
        html_map("reports/investigation_map.html", 600)
        st.caption("Use layer control (top-right) to toggle clusters, anomalies, routes, priority zones.")
    with mtabs[1]: html_map("reports/movement_heatmap.html", 600)
    with mtabs[2]: html_map("reports/cluster_map.html", 600)


# ═══════════════ TAB 7: EXPLAINABILITY ═══════════════
with tabs[6]:
    S("Explainability (XAI)", "Feature importance & prediction rationale")
    imp = pd.DataFrame({
        'Feature': ['Last_Longitude','Last_Latitude','Day','Time_Since_Last_Seen','Hour',
                     'Average_Speed','Average_Distance','Age_Group','Weather','Gender'],
        'Importance': [0.2872,0.2402,0.0983,0.0947,0.0803,0.0706,0.0441,0.0377,0.0289,0.0181]})
    fig = px.bar(imp,y='Feature',x='Importance',orientation='h',color='Importance',
                 color_continuous_scale='Blues',title="GradientBoosting -- Feature Importance")
    fig.update_layout(yaxis=dict(autorange='reversed'),height=400); P(fig)

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown('<div class="panel"><b style="color:#2563eb">Spatial -- 52.7%</b><br>Longitude + Latitude dominate predictions.</div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="panel"><b style="color:#f59e0b">Temporal -- 27.3%</b><br>Day, time, hour capture routines.</div>',unsafe_allow_html=True)
    with c3: st.markdown('<div class="panel"><b style="color:#22c55e">Kinematic -- 11.5%</b><br>Speed + distance constrain area reach.</div>',unsafe_allow_html=True)

    if os.path.exists("reports/prediction_explanation.txt"):
        S("Natural Language Explanation")
        with open("reports/prediction_explanation.txt",'r',encoding='utf-8') as f:
            st.markdown(f'<div class="panel" style="white-space:pre-wrap">{f.read()}</div>',unsafe_allow_html=True)


# ═══════════════ TAB 8: MODEL EVALUATION ═══════════════
with tabs[7]:
    S("Model Evaluation Dashboard", "Performance metrics & pipeline summary")
    if model_comp is not None:
        st.dataframe(model_comp,hide_index=True,use_container_width=True)
        S("Top-K Accuracy Curves")
        colors=['#2563eb','#22c55e','#f59e0b','#8b5cf6']
        fig=go.Figure()
        for i,(_,r) in enumerate(model_comp.iterrows()):
            fig.add_trace(go.Scatter(x=['Top-1','Top-3','Top-5'],
                y=[r['Top-1 Acc'],r['Top-3 Acc'],r['Top-5 Acc']],
                name=r['Model'],mode='lines+markers',line=dict(color=colors[i%4],width=3),marker=dict(size=10)))
        fig.update_layout(title="Top-K Accuracy",yaxis=dict(range=[0,1.05],title='Accuracy')); P(fig)

    S("Complete Pipeline Summary")
    pipeline=pd.DataFrame({
        'Stage':['Data Collection','Preprocessing','Feature Engineering','Clustering','Anomaly Detection',
                 'Case Generation','Location Prediction','Route Prediction','Search Priority','Explainability','Mapping'],
        'Status':['Complete']*11,
        'Output':['902,051 GPS points','896,818 cleaned','2,101 stay points','70->5 areas (Sil:0.446)',
                  '44,840 anomalies (5%)','112 cases (28 states)','4 models (GBM best)','3 routes (Markov)',
                  '3 regions ranked','Feature importance','3 interactive maps'],
        'ML Technique':['Synthetic Gen','Speed Filter','Haversine+Stay','DBSCAN+K-Means','IF+LOF+SVM',
                        'Profile Sampling','RF/XGB/GBM/KNN','Markov+Beam','6-Factor Composite',
                        'SHAP/Tree Imp','Folium Layers']})
    st.dataframe(pipeline,hide_index=True,use_container_width=True)

    st.markdown(f"""<div class="panel" style="margin-top:1rem">
        <b style="color:#2563eb">CASEFILE</b> Pan-India Missing Person System -- Academic Simulation covering
        all 28 States and 8 Union Territories of India.<br>
        <span style="color:#94a3b8">Built with: Python 3.14 | Streamlit | scikit-learn | XGBoost | Folium | Plotly</span>
    </div>""", unsafe_allow_html=True)
