import streamlit as st


def apply_custom_css():
    """Inject modern glassmorphism CSS styles into Streamlit."""
    st.markdown("""
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Glassmorphism Metric Card */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .metric-label {
            font-size: 0.85rem;
            color: #94A3B8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #F8FAFC;
        }

        .metric-delta-positive {
            font-size: 0.9rem;
            font-weight: 600;
            color: #10B981;
        }

        .metric-delta-negative {
            font-size: 0.9rem;
            font-weight: 600;
            color: #EF4444;
        }

        /* Custom Header Styling */
        .header-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }

        .header-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Reduce Streamlit default top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, is_positive: bool = True):
    """Render a custom styled metric card."""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-positive" if is_positive else "metric-delta-negative"
        delta_html = f'<div class="{delta_class}">{delta}</div>'

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)