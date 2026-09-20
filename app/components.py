import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.graph_objects as go


def metric_card(label, value, icon="", delta=None, color="#1f77b4"):
    """Styled metric card with icon and custom color accent."""
    delta_html = ""
    if delta:
        delta_html = f'<p style="font-size:0.8rem; color:#21c354; margin:0;">▲ {delta}</p>'
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
        <p style="font-size:0.75rem; color:#8899a6; margin:0; text-transform:uppercase; letter-spacing:1px;">{icon} {label}</p>
        <p style="font-size:1.6rem; font-weight:700; color:#e8eaed; margin:0.3rem 0 0 0;">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def status_badge(text, level="info"):
    """Colored status badge."""
    colors = {
        "critical": ("#ff4b4b", "#2d0a0a"),
        "warning": ("#ffa421", "#2d1f0a"),
        "success": ("#21c354", "#0a2d12"),
        "info": ("#4b8bff", "#0a152d"),
    }
    fg, bg = colors.get(level, colors["info"])
    return f'<span style="background:{bg}; color:{fg}; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid {fg}40;">{text}</span>'


def section_header(title, subtitle=""):
    """Section header with underline."""
    sub = f'<span style="color:#8899a6; font-size:0.85rem;">{subtitle}</span><br>' if subtitle else ""
    st.markdown(f'{sub}<span style="color:#e8eaed; font-size:1.3rem; font-weight:600; border-bottom: 2px solid #4b8bff; padding-bottom:6px;">{title}</span>', unsafe_allow_html=True)
    st.markdown("")


def priority_badge(priority):
    """Priority level badge."""
    p = str(priority).upper()
    if "VERY HIGH" in p:
        return status_badge("⬤ VERY HIGH", "critical")
    elif "HIGH" in p:
        return status_badge("⬤ HIGH", "warning")
    elif "MEDIUM" in p:
        return status_badge("⬤ MEDIUM", "info")
    else:
        return status_badge("⬤ LOW", "success")


def priority_color(priority):
    p = str(priority).upper()
    if 'VERY HIGH' in p:
        return '#ff4b4b'
    elif 'HIGH' in p:
        return '#ffa421'
    elif 'MEDIUM' in p:
        return '#4b8bff'
    return '#21c354'


def styled_dataframe(df):
    """Apply custom styling to dataframe."""
    if 'priority' in [c.lower() for c in df.columns]:
        col = [c for c in df.columns if c.lower() == 'priority'][0]

        def color_priority(val):
            val = str(val).upper()
            if 'VERY HIGH' in val:
                return 'color: #ff4b4b; font-weight: bold'
            elif 'HIGH' in val:
                return 'color: #ffa421; font-weight: bold'
            elif 'MEDIUM' in val:
                return 'color: #4b8bff'
            return 'color: #21c354'

        return df.style.map(color_priority, subset=[col])
    return df


def create_mini_map(lat, lon, zoom=13):
    """Minimal folium map with marker."""
    m = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri English', max_zoom=18
    ).add_to(m)
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color='red', icon='exclamation-sign'),
        popup=f"Last Known: {lat:.4f}, {lon:.4f}"
    ).add_to(m)
    folium.Circle(
        [lat, lon], radius=500, color='#ff4b4b', fill=True, fill_opacity=0.15,
        popup="500m Search Radius"
    ).add_to(m)
    return m


def radar_chart(categories, values, title=""):
    """Create a radar/spider chart."""
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(75, 139, 255, 0.2)',
        line=dict(color='#4b8bff', width=2),
        marker=dict(size=6, color='#4b8bff')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                            gridcolor='#333', tickfont=dict(color='#888')),
            angularaxis=dict(gridcolor='#333', tickfont=dict(color='#ccc', size=11))
        ),
        showlegend=False,
        title=dict(text=title, font=dict(color='#e8eaed', size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=40, b=40),
        height=350
    )
    return fig
