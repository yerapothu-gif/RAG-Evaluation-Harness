"""
Baahubali Royal Theme for Streamlit.
Complete CSS override for a cinematic, premium look.
"""
from src.config import ROYAL_GOLD, DEEP_MAROON, DARK_BROWN, WARM_CREAM, MUTED_GOLD, EMBER_ORANGE


def get_custom_css(dark_mode: bool = True) -> str:
    """Generate custom CSS based on theme mode."""
    if dark_mode:
        bg_primary = "#0E0A07"
        bg_secondary = "#1A120D"
        bg_card = "#1E1510"
        text_primary = WARM_CREAM
        text_secondary = "#B8A990"
        border_color = "rgba(212, 175, 55, 0.2)"
        shadow_color = "rgba(212, 175, 55, 0.05)"
    else:
        bg_primary = "#F5F0E8"
        bg_secondary = "#EDE5D8"
        bg_card = "#FFFFFF"
        text_primary = "#1A120D"
        text_secondary = "#4A3C2E"
        border_color = "rgba(212, 175, 55, 0.3)"
        shadow_color = "rgba(0, 0, 0, 0.08)"

    return f"""
    <style>
        /* === Google Fonts === */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cinzel:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

        /* === Global Reset === */
        .stApp {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        /* === Sidebar === */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {bg_secondary} 0%, {bg_primary} 100%) !important;
            border-right: 1px solid {border_color} !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            font-family: 'Cinzel', serif !important;
            color: {ROYAL_GOLD} !important;
        }}

        /* === Headers === */
        h1, .stMarkdown h1 {{
            font-family: 'Cinzel Decorative', serif !important;
            color: {ROYAL_GOLD} !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
            text-shadow: 0 0 20px rgba(212, 175, 55, 0.15) !important;
        }}

        h2, h3, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'Cinzel', serif !important;
            color: {ROYAL_GOLD} !important;
            font-weight: 600 !important;
        }}

        /* === Regular Text === */
        p, span, label, .stMarkdown p {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_primary} !important;
        }}

        /* === Card Containers === */
        div[data-testid="stExpander"] {{
            background: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        /* === Metrics === */
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {bg_card} 0%, {bg_secondary} 100%) !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 16px {shadow_color} !important;
        }}

        div[data-testid="stMetric"] label {{
            color: {text_secondary} !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            font-size: 0.75rem !important;
            letter-spacing: 1px !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {ROYAL_GOLD} !important;
            font-family: 'Cinzel', serif !important;
            font-weight: 700 !important;
        }}

        /* === Buttons === */
        .stButton > button {{
            background: linear-gradient(135deg, {ROYAL_GOLD} 0%, {MUTED_GOLD} 100%) !important;
            color: #0E0A07 !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Cinzel', serif !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            padding: 8px 24px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2) !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(212, 175, 55, 0.35) !important;
        }}

        /* === Text Input === */
        .stTextInput > div > div > input {{
            background: {bg_card} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            font-family: 'Outfit', sans-serif !important;
            padding: 12px 16px !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {ROYAL_GOLD} !important;
            box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2) !important;
        }}

        /* === Tabs === */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px !important;
            background: {bg_secondary} !important;
            border-radius: 10px !important;
            padding: 4px !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            font-family: 'Cinzel', serif !important;
            font-weight: 500 !important;
            color: {text_secondary} !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(212, 175, 55, 0.05) 100%) !important;
            color: {ROYAL_GOLD} !important;
            border-bottom: 2px solid {ROYAL_GOLD} !important;
        }}

        /* === Progress Bar === */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {ROYAL_GOLD} 0%, {EMBER_ORANGE} 100%) !important;
        }}

        /* === Selectbox === */
        .stSelectbox > div > div {{
            background: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
        }}

        /* === Divider === */
        hr {{
            border-color: {border_color} !important;
        }}

        /* === Ornamental Divider === */
        .ornament-divider {{
            text-align: center;
            color: {ROYAL_GOLD};
            font-size: 1.2rem;
            margin: 20px 0;
            opacity: 0.5;
        }}

        /* === Custom Card === */
        .royal-card {{
            background: linear-gradient(135deg, {bg_card} 0%, {bg_secondary} 100%);
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 16px {shadow_color};
        }}

        /* === Pill/Badge === */
        .category-pill {{
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            letter-spacing: 0.5px;
        }}

        .pill-character {{ background: rgba(212, 175, 55, 0.15); color: {ROYAL_GOLD}; border: 1px solid rgba(212, 175, 55, 0.3); }}
        .pill-kingdom {{ background: rgba(45, 90, 61, 0.15); color: #5DAE7A; border: 1px solid rgba(45, 90, 61, 0.3); }}
        .pill-battle {{ background: rgba(139, 26, 26, 0.15); color: #D45B5B; border: 1px solid rgba(139, 26, 26, 0.3); }}
        .pill-timeline {{ background: rgba(100, 130, 180, 0.15); color: #7BA3D4; border: 1px solid rgba(100, 130, 180, 0.3); }}
        .pill-general {{ background: rgba(184, 169, 144, 0.15); color: #B8A990; border: 1px solid rgba(184, 169, 144, 0.3); }}

        .pill-inscope {{ background: rgba(45, 90, 61, 0.2); color: #5DAE7A; border: 1px solid rgba(45, 90, 61, 0.3); }}
        .pill-outscope {{ background: rgba(139, 26, 26, 0.2); color: #D45B5B; border: 1px solid rgba(139, 26, 26, 0.3); }}

        /* === Scrollbar === */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {bg_primary};
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(212, 175, 55, 0.3);
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(212, 175, 55, 0.5);
        }}

        /* === Radio Buttons === */
        .stRadio > div {{
            gap: 8px !important;
        }}

        /* === Toast/Alert === */
        .stAlert {{
            border-radius: 10px !important;
            border-left: 4px solid {ROYAL_GOLD} !important;
        }}

        /* === DataFrame/Table === */
        .stDataFrame {{
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }}

        /* === Animated Glow Text === */
        @keyframes goldGlow {{
            0% {{ text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); }}
            50% {{ text-shadow: 0 0 20px rgba(212, 175, 55, 0.4), 0 0 40px rgba(212, 175, 55, 0.1); }}
            100% {{ text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); }}
        }}

        .glow-text {{
            animation: goldGlow 3s ease-in-out infinite;
        }}

        /* === Hide Streamlit Branding while preserving Sidebar Toggle === */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{
            background-color: transparent !important;
        }}
        button[data-testid="stSidebarToggle"],
        div[data-testid="stSidebarNavCollapseButton"],
        div[data-testid="stHeader"] button {{
            visibility: visible !important;
            display: flex !important;
            color: {ROYAL_GOLD} !important;
            z-index: 999999 !important;
        }}
    </style>

    """


def render_ornament():
    """Return an ornamental divider HTML."""
    return '<div class="ornament-divider">━━━ ⚜ ━━━</div>'


def render_category_pill(category: str) -> str:
    """Render a styled category pill badge."""
    css_class = f"pill-{category.lower()}"
    icon = {"Character": "👤", "Kingdom": "🏰", "Battle": "⚔️", "Timeline": "📅", "General": "📜"}.get(category, "❓")
    return f'<span class="category-pill {css_class}">{icon} {category}</span>'


def render_scope_pill(is_in_scope: bool) -> str:
    """Render an in-scope/out-of-scope pill badge."""
    if is_in_scope:
        return '<span class="category-pill pill-inscope">✅ In-Scope</span>'
    else:
        return '<span class="category-pill pill-outscope">❌ Out-of-Scope</span>'
