import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Must be FIRST Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_model()

# ── PREMIUM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #050810;
    --surface: #0c1220;
    --border:  rgba(255,255,255,0.06);
    --accent:  #00d4ff;
    --accent2: #7c3aed;
    --danger:  #ff4d6d;
    --safe:    #00f5a0;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 64px 64px;
    pointer-events: none; z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 700px 500px at 5% 10%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 600px 500px at 95% 90%, rgba(124,58,237,0.07) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSlider label p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

[data-testid="stSlider"] p {
    color: var(--accent) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    border: 2px solid #050810 !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.55) !important;
    width: 18px !important; height: 18px !important;
    transition: all 0.2s !important;
}
[data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 28px rgba(0,212,255,0.85) !important;
    transform: scale(1.25) !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.82rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 15px !important;
    cursor: pointer !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.25) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 55px rgba(0,212,255,0.55) !important;
}

#MainMenu, footer, header { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 2px; }

hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 1.5rem 0 !important;
}

h1,h2,h3 {
    font-family: 'Syne', sans-serif !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
             background:linear-gradient(90deg,#00d4ff,#7c3aed);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🩺 DiabetesAI
        </div>
        <div style="color:#334155;font-size:0.73rem;margin-top:3px;letter-spacing:1px;
             text-transform:uppercase;">Risk Intelligence</div>
    </div>
    <hr>
    <div style="color:#475569;font-size:0.75rem;font-weight:500;
         letter-spacing:0.5px;margin-bottom:1.2rem;text-transform:uppercase;">
        📋 Clinical Inputs
    </div>
    """, unsafe_allow_html=True)

    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose     = st.slider("Glucose (mg/dL)", 0, 199, 120)
    bp          = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    skin        = st.slider("Skin Thickness (mm)", 0, 99, 25)
    insulin     = st.slider("Insulin (IU/mL)", 0, 846, 80)
    bmi         = st.slider("BMI", 0.0, 67.1, 25.0, 0.1)
    dpf         = st.slider("Diabetes Pedigree", 0.078, 2.42, 0.50, 0.001)
    age         = st.slider("Age (years)", 21, 81, 33)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔬 Analyze Risk Now", use_container_width=True)
    if analyze_clicked:
        st.toast("✅ Analysis complete! Results updated below.", icon="🔬")

    st.markdown("""
    <div style="margin-top:1.2rem;padding:14px 16px;
         background:rgba(0,212,255,0.04);
         border:1px solid rgba(0,212,255,0.1);border-radius:10px;">
        <div style="color:#64748b;font-size:0.68rem;line-height:1.8;letter-spacing:0.3px;">
            ⚡ <b style="color:#94a3b8;">Model:</b> Random Forest<br>
            📊 <b style="color:#94a3b8;">Accuracy:</b> 74.03%<br>
            👥 <b style="color:#94a3b8;">Trained on:</b> 768 patients<br>
            🔒 Educational use only
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PREDICTION ────────────────────────────────────────────────────────────────
input_data   = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0]
risk_pct     = round(probability[1] * 100, 1)
safe_pct     = round(probability[0] * 100, 1)
is_high      = prediction == 1

rc     = "#ff4d6d" if is_high else "#00f5a0"
rb     = "rgba(255,77,109,0.07)" if is_high else "rgba(0,245,160,0.07)"
rb2    = "rgba(255,77,109,0.2)"  if is_high else "rgba(0,245,160,0.2)"
rtitle = "⚠ HIGH RISK DETECTED"  if is_high else "✓ LOW RISK PROFILE"
rdesc  = ("Elevated biomarker levels indicate significant diabetes risk. "
          "Immediate specialist consultation is recommended." if is_high else
          "Biomarker profile suggests low diabetes risk. "
          "Maintain healthy habits and schedule routine checkups.")


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:2rem 0 1.5rem 0;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.18);
             border-radius:20px;padding:5px 14px;font-size:0.68rem;font-weight:600;
             letter-spacing:2px;text-transform:uppercase;color:#00d4ff;">
            AI Clinical Decision Support
        </div>
        <div style="background:{rb2};border:1px solid {rc}44;
             border-radius:20px;padding:5px 14px;font-size:0.68rem;font-weight:700;
             letter-spacing:1px;color:{rc};">
            {rtitle}
        </div>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
         line-height:1.1;letter-spacing:-1.5px;">
        Diabetes Risk
        <span style="background:linear-gradient(135deg,#00d4ff,#7c3aed);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Intelligence
        </span>
    </div>
    <div style="color:#334155;font-size:0.85rem;margin-top:0.5rem;max-width:500px;line-height:1.5;">
        Machine learning biomarker analysis for early diabetes risk stratification
    </div>
</div>
""", unsafe_allow_html=True)


# ── STAT ROW ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;">
    <div style="background:#0c1220;border:1px solid rgba(255,255,255,0.05);
         border-top:2px solid #00d4ff;border-radius:12px;padding:16px 12px;overflow:hidden;">
        <div style="font-size:1rem;margin-bottom:8px;">🤖</div>
        <div style="font-size:0.55rem;letter-spacing:1.5px;text-transform:uppercase;
             color:#475569;margin-bottom:4px;">Model</div>
        <div style="font-family:'Syne',sans-serif;font-size:0.78rem;
             font-weight:800;color:#00d4ff;overflow:hidden;text-overflow:ellipsis;
             white-space:nowrap;">Random Forest</div>
    </div>
    <div style="background:#0c1220;border:1px solid rgba(255,255,255,0.05);
         border-top:2px solid #7c3aed;border-radius:12px;padding:16px 12px;overflow:hidden;">
        <div style="font-size:1rem;margin-bottom:8px;">🎯</div>
        <div style="font-size:0.55rem;letter-spacing:1.5px;text-transform:uppercase;
             color:#475569;margin-bottom:4px;">Accuracy</div>
        <div style="font-family:'Syne',sans-serif;font-size:0.78rem;
             font-weight:800;color:#7c3aed;white-space:nowrap;">74.03%</div>
    </div>
    <div style="background:#0c1220;border:1px solid rgba(255,255,255,0.05);
         border-top:2px solid #00f5a0;border-radius:12px;padding:16px 12px;overflow:hidden;">
        <div style="font-size:1rem;margin-bottom:8px;">👥</div>
        <div style="font-size:0.55rem;letter-spacing:1.5px;text-transform:uppercase;
             color:#475569;margin-bottom:4px;">Dataset</div>
        <div style="font-family:'Syne',sans-serif;font-size:0.78rem;
             font-weight:800;color:#00f5a0;white-space:nowrap;">768 Patients</div>
    </div>
    <div style="background:#0c1220;border:1px solid rgba(255,255,255,0.05);
         border-top:2px solid #ff9f43;border-radius:12px;padding:16px 12px;overflow:hidden;">
        <div style="font-size:1rem;margin-bottom:8px;">🔬</div>
        <div style="font-size:0.55rem;letter-spacing:1.5px;text-transform:uppercase;
             color:#475569;margin-bottom:4px;">Features</div>
        <div style="font-family:'Syne',sans-serif;font-size:0.78rem;
             font-weight:800;color:#ff9f43;white-space:nowrap;">8 Biomarkers</div>
    </div>
</div>
""", unsafe_allow_html=True)



# ── MAIN TWO COLUMNS ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    # Result card
    st.markdown(f"""
    <div style="background:{rb};border:1px solid {rc}33;border-left:4px solid {rc};
         border-radius:14px;padding:26px;margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1rem;
             font-weight:800;color:{rc};margin-bottom:0.6rem;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{rtitle}</div>
        <div style="color:#64748b;font-size:0.83rem;line-height:1.6;
             margin-bottom:1.5rem;">{rdesc}</div>
        <div style="display:flex;gap:2rem;">
            <div>
                <div style="font-size:0.62rem;letter-spacing:2px;
                     text-transform:uppercase;color:#334155;margin-bottom:3px;">
                     Risk Score</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.9rem;
                     font-weight:800;color:{rc};line-height:1;">{risk_pct}%</div>
            </div>
            <div>
                <div style="font-size:0.62rem;letter-spacing:2px;
                     text-transform:uppercase;color:#334155;margin-bottom:3px;">
                     Safe Score</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.9rem;
                     font-weight:800;color:#00f5a0;line-height:1;">{safe_pct}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': "%", 'font': {'size': 36, 'color': rc, 'family': 'Syne'}},
        gauge={
            'axis': {'range': [0,100], 'tickcolor':'#1e293b',
                     'tickfont':{'color':'#334155','size':9}},
            'bar': {'color': rc, 'thickness': 0.28},
            'bgcolor': '#0c1220',
            'borderwidth': 0,
            'steps': [
                {'range':[0,30],   'color':'rgba(0,245,160,0.08)'},
                {'range':[30,60],  'color':'rgba(255,159,67,0.08)'},
                {'range':[60,100], 'color':'rgba(255,77,109,0.08)'}
            ],
            'threshold': {'line':{'color':rc,'width':3},'thickness':0.75,'value':risk_pct}
        },
        title={'text':'RISK METER','font':{'size':10,'color':'#334155','family':'DM Sans'}}
    ))
    fig_g.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=210, margin=dict(l=20,r=20,t=35,b=0)
    )
    st.plotly_chart(fig_g, use_container_width=True, config={'displayModeBar':False})

    # Horizontal bar
    fig_hb = go.Figure(go.Bar(
        x=[risk_pct, safe_pct],
        y=["Diabetes Risk", "No Diabetes"],
        orientation='h',
        marker=dict(color=[rc, "#00f5a0"], line=dict(width=0)),
        text=[f"{risk_pct}%", f"{safe_pct}%"],
        textposition='inside',
        textfont=dict(color='white', size=13, family='Syne'),
        width=0.45
    ))
    fig_hb.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#64748b', size=12),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0,108]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#64748b', size=12)),
        margin=dict(l=0,r=0,t=8,b=0), height=105, showlegend=False
    )
    st.plotly_chart(fig_hb, use_container_width=True, config={'displayModeBar':False})

with right:
    # Donut
    fig_d = go.Figure(go.Pie(
        labels=["Diabetes Risk","No Diabetes"],
        values=[risk_pct, safe_pct],
        hole=0.66,
        marker=dict(colors=[rc,"#00f5a0"], line=dict(color='#050810',width=4)),
        textinfo='percent',
        textfont=dict(color='white', size=13, family='Syne'),
        hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>"
    ))
    fig_d.add_annotation(
        text=f"<b>{risk_pct}%</b>",
        x=0.5, y=0.55, showarrow=False,
        font=dict(size=26, color=rc, family='Syne')
    )
    fig_d.add_annotation(
        text="RISK",
        x=0.5, y=0.38, showarrow=False,
        font=dict(size=10, color='#334155', family='DM Sans')
    )
    fig_d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#64748b'),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.05,
                    font=dict(color='#64748b',size=11), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0,r=0,t=10,b=30), height=240
    )
    st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar':False})

    # Feature importance
    feat_labels = ['Glucose','BMI','Age','Pedigree','BloodPressure',
                   'Pregnancies','Insulin','SkinThickness']
    try:
        imp = model.feature_importances_
        idx = np.argsort(imp)
        sf  = [feat_labels[i] if i < len(feat_labels) else f"F{i}" for i in idx]
        sv  = imp[idx]
    except:
        sf  = feat_labels
        sv  = [0.28,0.15,0.13,0.12,0.10,0.09,0.08,0.05]

    bar_colors = ['#7c3aed' if v == max(sv) else
                  '#00d4ff' if v > np.mean(sv) else
                  '#1e3a5f' for v in sv]

    fig_i = go.Figure(go.Bar(
        x=sv, y=sf, orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in sv],
        textposition='outside',
        textfont=dict(color='#475569', size=10, family='DM Sans'),
        width=0.6
    ))
    fig_i.update_layout(
        title=dict(text="Feature Importance",
                   font=dict(size=11,color='#475569',family='DM Sans'), x=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#64748b', size=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                   tickfont=dict(color='#334155'), zeroline=False,
                   range=[0, max(sv)*1.3]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11)),
        margin=dict(l=0,r=55,t=28,b=0), height=255, showlegend=False, bargap=0.35
    )
    st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar':False})


# ── BIOMARKER CARDS ───────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;
     letter-spacing:2px;text-transform:uppercase;color:#334155;
     margin:1.5rem 0 1rem 0;">◈ Patient Biomarker Profile</div>
""", unsafe_allow_html=True)

bm_labels   = ["Pregnancies","Glucose","Blood Pressure","Skin Thickness",
               "Insulin","BMI","Pedigree Fn","Age"]
bm_values   = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
bm_units    = ["count","mg/dL","mm Hg","mm","IU/mL","kg/m²","score","years"]
bm_normals  = ["0–3","<140","<80","<30","<166","18.5–25","<0.5","–"]
bm_icons    = ["🤰","🩸","💗","📏","💉","⚖️","🧬","🎂"]

def bm_status(i, v):
    if i==1: return ("⚠ High","#ff4d6d") if v>=140 else ("✓ OK","#00f5a0")
    if i==2: return ("⚠ High","#ff4d6d") if v>=80  else ("✓ OK","#00f5a0")
    if i==5: return ("⚠ Obese","#ff4d6d") if v>=30 else ("✓ OK","#00f5a0")
    return ("✓ OK","#00f5a0")

cols = st.columns(4)
for i,(lbl,val,unit,norm,icon) in enumerate(zip(bm_labels,bm_values,bm_units,bm_normals,bm_icons)):
    stat, clr = bm_status(i, val)
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:#0c1220;border:1px solid rgba(255,255,255,0.05);
             border-radius:12px;padding:16px;margin-bottom:10px;
             border-left:3px solid {clr};">
            <div style="display:flex;justify-content:space-between;align-items:center;
                 margin-bottom:8px;">
                <span style="font-size:1.1rem;">{icon}</span>
                <span style="font-size:0.65rem;font-weight:700;color:{clr};
                      background:{clr}18;padding:2px 8px;border-radius:20px;">{stat}</span>
            </div>
            <div style="font-size:0.62rem;letter-spacing:1.5px;text-transform:uppercase;
                 color:#334155;margin-bottom:3px;">{lbl}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;
                 font-weight:800;color:{clr};line-height:1.1;">{val}
                 <span style="font-size:0.7rem;font-weight:400;color:#334155;">{unit}</span>
            </div>
            <div style="font-size:0.65rem;color:#1e3a5f;margin-top:4px;">
                Normal: {norm}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RADAR CHART ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;
     letter-spacing:2px;text-transform:uppercase;color:#334155;
     margin:1.5rem 0 1rem 0;">◈ Biomarker Radar Analysis</div>
""", unsafe_allow_html=True)

max_v  = [17,199,122,99,846,67.1,2.42,81]
r_vals = [min(v/m,1.0) for v,m in zip(bm_values, max_v)]
r_lbls = ["Pregnancies","Glucose","Blood Pressure","Skin","Insulin","BMI","Pedigree","Age"]

fig_r = go.Figure()
fig_r.add_trace(go.Scatterpolar(
    r=r_vals+[r_vals[0]], theta=r_lbls+[r_lbls[0]],
    fill='toself',
    fillcolor=f'rgba({255 if is_high else 0},{77 if is_high else 245},{109 if is_high else 160},0.1)',
    line=dict(color=rc, width=2.5),
    marker=dict(color=rc, size=6),
    name="Patient"
))
fig_r.update_layout(
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(visible=True,range=[0,1],
                        tickfont=dict(color='#1e293b',size=8),
                        gridcolor='rgba(255,255,255,0.05)',
                        linecolor='rgba(255,255,255,0.05)'),
        angularaxis=dict(tickfont=dict(color='#64748b',size=12),
                         gridcolor='rgba(255,255,255,0.05)',
                         linecolor='rgba(255,255,255,0.07)')
    ),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#64748b'),
    showlegend=False,
    margin=dict(l=50,r=50,t=20,b=20), height=420
)
st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar':False})


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.07);
     padding:2.5rem 0 1.5rem 0;text-align:center;margin-top:2rem;">
    <div style="font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;
         letter-spacing:3px;text-transform:uppercase;color:#64748b;margin-bottom:0.6rem;">
        ⚕ Medical Disclaimer
    </div>
    <div style="color:#475569;font-size:0.78rem;max-width:520px;
         margin:0 auto;line-height:1.8;">
        This tool is for educational purposes only and does not constitute medical advice.
        Always consult a qualified healthcare professional for diagnosis and treatment decisions.
    </div>
    <div style="color:#334155;font-size:0.7rem;margin-top:1.2rem;letter-spacing:1.5px;">
        Built by <span style="color:#00d4ff;font-weight:600;">Zafii ML</span>
        &nbsp;·&nbsp; Random Forest Classifier
        &nbsp;·&nbsp; Powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)
