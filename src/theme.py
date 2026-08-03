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
        bg_glass = "rgba(30, 21, 16, 0.85)"
        text_primary = WARM_CREAM
        text_secondary = "#B8A990"
        border_color = "rgba(212, 175, 55, 0.2)"
        border_hover = "rgba(212, 175, 55, 0.5)"
        shadow_color = "rgba(212, 175, 55, 0.05)"
        shadow_hover = "rgba(212, 175, 55, 0.15)"
        input_bg = "#231A12"
        nav_active_bg = "rgba(212, 175, 55, 0.12)"
        nav_hover_bg = "rgba(212, 175, 55, 0.06)"
    else:
        bg_primary = "#F5F0E8"
        bg_secondary = "#EDE5D8"
        bg_card = "#FFFFFF"
        bg_glass = "rgba(255, 255, 255, 0.9)"
        text_primary = "#1A120D"
        text_secondary = "#4A3C2E"
        border_color = "rgba(212, 175, 55, 0.3)"
        border_hover = "rgba(212, 175, 55, 0.6)"
        shadow_color = "rgba(0, 0, 0, 0.08)"
        shadow_hover = "rgba(212, 175, 55, 0.25)"
        input_bg = "#FDFAF5"
        nav_active_bg = "rgba(212, 175, 55, 0.15)"
        nav_hover_bg = "rgba(212, 175, 55, 0.08)"

    return f"""
    <style>
        /* === Google Fonts === */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cinzel:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

        /* === Material Icons Class === */
        .msi {{
            font-family: 'Material Symbols Rounded', sans-serif !important;
            font-weight: normal;
            font-style: normal;
            font-size: 1.1em;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none !important;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            vertical-align: -3px;
            -webkit-font-smoothing: antialiased;
            margin-right: 5px;
        }}
        /* Filled variant */
        .msi-fill {{
            font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }}
        /* Status dot via CSS — no emoji */
        .status-dot {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
            position: relative; top: -1px;
        }}
        .status-dot-active  {{ background: #5DAE7A; box-shadow: 0 0 5px rgba(93,174,122,0.6); }}
        .status-dot-pending {{ background: #D4A84B; box-shadow: 0 0 5px rgba(212,168,75,0.5); }}
        .status-dot-offline {{ background: #888; }}

        /* ============================================================
           GLOBAL RESET & BASE
        ============================================================ */
        .stApp {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        /* Subtle background grain texture */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
            pointer-events: none;
            z-index: 0;
        }}

        /* ============================================================
           SIDEBAR
        ============================================================ */
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

        /* Sidebar Brand Block */
        .sidebar-brand {{
            text-align: center;
            padding: 8px 4px 4px 4px;
        }}
        .sidebar-brand-title {{
            font-family: 'Cinzel Decorative', serif !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: {ROYAL_GOLD} !important;
            letter-spacing: 2px !important;
            line-height: 1.3 !important;
            text-shadow: 0 0 20px rgba(212, 175, 55, 0.3) !important;
            margin: 0 !important;
        }}
        .sidebar-brand-subtitle {{
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.7rem !important;
            font-weight: 400 !important;
            color: {text_secondary} !important;
            letter-spacing: 3px !important;
            text-transform: uppercase !important;
            margin: 4px 0 0 0 !important;
        }}

        /* Sidebar Theme Toggle */
        .theme-toggle-container {{
            display: flex;
            gap: 8px;
            margin: 12px 0;
            padding: 6px;
            background: {bg_secondary};
            border: 1px solid {border_color};
            border-radius: 12px;
        }}
        .theme-btn {{
            flex: 1;
            padding: 7px 12px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-family: 'Outfit', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.25s ease;
            text-align: center;
        }}
        .theme-btn-active {{
            background: linear-gradient(135deg, {ROYAL_GOLD}, {MUTED_GOLD});
            color: #0E0A07;
            box-shadow: 0 2px 8px rgba(212, 175, 55, 0.35);
        }}
        .theme-btn-inactive {{
            background: transparent;
            color: {text_secondary};
        }}

        /* Navigation Radio Buttons */
        .stRadio > label {{
            display: none !important;
        }}
        .stRadio > div {{
            gap: 4px !important;
            flex-direction: column !important;
        }}
        .stRadio > div > label {{
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }}
        .stRadio > div > label:hover {{
            background: {nav_hover_bg} !important;
            border-color: {border_color} !important;
        }}
        .stRadio > div > label[data-baseweb] span {{
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.9rem !important;
            color: {text_secondary} !important;
        }}

        /* ============================================================
           HEADERS & TYPOGRAPHY
        ============================================================ */
        h1, .stMarkdown h1 {{
            font-family: 'Cinzel Decorative', serif !important;
            color: {ROYAL_GOLD} !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
            text-shadow: 0 0 30px rgba(212, 175, 55, 0.2) !important;
            margin-bottom: 4px !important;
        }}

        h2, h3, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'Cinzel', serif !important;
            color: {ROYAL_GOLD} !important;
            font-weight: 600 !important;
        }}

        h4, h5, .stMarkdown h4, .stMarkdown h5 {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_secondary} !important;
            font-weight: 500 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            font-size: 0.8rem !important;
        }}

        p, label, .stMarkdown p {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_primary} !important;
            line-height: 1.7 !important;
        }}

        /* Page subtitle style */
        .page-subtitle {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_secondary} !important;
            font-size: 0.95rem !important;
            font-style: italic !important;
            margin-top: -4px !important;
            margin-bottom: 8px !important;
        }}

        /* ============================================================
           BUTTONS
        ============================================================ */
        .stButton > button {{
            background: linear-gradient(135deg, {ROYAL_GOLD} 0%, {MUTED_GOLD} 100%) !important;
            color: #0E0A07 !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'Cinzel', serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.8px !important;
            padding: 10px 24px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 14px rgba(212, 175, 55, 0.25) !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(212, 175, 55, 0.4) !important;
            filter: brightness(1.08) !important;
        }}

        .stButton > button:active {{
            transform: translateY(0px) !important;
            box-shadow: 0 3px 10px rgba(212, 175, 55, 0.3) !important;
        }}

        /* Secondary buttons */
        .stButton > button[kind="secondary"] {{
            background: transparent !important;
            color: {ROYAL_GOLD} !important;
            border: 1px solid {border_color} !important;
            box-shadow: none !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            background: {nav_hover_bg} !important;
            border-color: {border_hover} !important;
            transform: translateY(-1px) !important;
        }}

        /* Form submit button */
        .stFormSubmitButton > button {{
            background: linear-gradient(135deg, {ROYAL_GOLD} 0%, {EMBER_ORANGE} 100%) !important;
            color: #0E0A07 !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'Cinzel', serif !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            font-size: 1rem !important;
            padding: 12px 32px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 6px 20px rgba(212, 175, 55, 0.35) !important;
        }}
        .stFormSubmitButton > button:hover {{
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 30px rgba(212, 175, 55, 0.5) !important;
            filter: brightness(1.1) !important;
        }}

        /* ============================================================
           TEXT INPUT & FORM
        ============================================================ */
        .stTextInput > div > div > input {{
            background: {input_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1rem !important;
            padding: 14px 18px !important;
            transition: all 0.25s ease !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {ROYAL_GOLD} !important;
            box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.15) !important;
            outline: none !important;
        }}

        .stTextInput > div > div > input::placeholder {{
            color: {text_secondary} !important;
            opacity: 0.6 !important;
        }}

        .stTextInput > label {{
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            color: {text_secondary} !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.5px !important;
        }}

        /* Password input */
        .stTextInput > div > div > div > input {{
            background: {input_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            color: {text_primary} !important;
        }}

        /* ============================================================
           TABS
        ============================================================ */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px !important;
            background: {bg_secondary} !important;
            border-radius: 12px !important;
            padding: 5px !important;
            border: 1px solid {border_color} !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            font-family: 'Cinzel', serif !important;
            font-weight: 500 !important;
            color: {text_secondary} !important;
            border-radius: 9px !important;
            padding: 10px 22px !important;
            transition: all 0.2s ease !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: {ROYAL_GOLD} !important;
            background: {nav_hover_bg} !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.18) 0%, rgba(212, 175, 55, 0.08) 100%) !important;
            color: {ROYAL_GOLD} !important;
            border-bottom: 2px solid {ROYAL_GOLD} !important;
            font-weight: 600 !important;
        }}

        /* ============================================================
           EXPANDERS / CARDS
        ============================================================ */
        div[data-testid="stExpander"] {{
            background: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            margin-bottom: 12px !important;
            transition: border-color 0.2s ease !important;
        }}

        div[data-testid="stExpander"]:hover {{
            border-color: {border_hover} !important;
        }}

        div[data-testid="stExpander"] summary {{
            padding: 14px 18px !important;
        }}

        div[data-testid="stExpander"] summary p {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_primary} !important;
            font-weight: 500 !important;
        }}

        /* Protect Streamlit icon glyphs from font corruption */
        [data-testid="stIcon"],
        div[data-testid="stExpander"] summary span[class*="icon"],
        div[data-testid="stExpander"] summary i {{
            font-family: inherit !important;
        }}

        /* ============================================================
           METRICS
        ============================================================ */
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {bg_card} 0%, {bg_secondary} 100%) !important;
            border: 1px solid {border_color} !important;
            border-radius: 14px !important;
            padding: 18px 22px !important;
            box-shadow: 0 4px 16px {shadow_color} !important;
            transition: all 0.25s ease !important;
        }}

        div[data-testid="stMetric"]:hover {{
            border-color: {border_hover} !important;
            box-shadow: 0 8px 24px {shadow_hover} !important;
            transform: translateY(-2px) !important;
        }}

        div[data-testid="stMetric"] label {{
            color: {text_secondary} !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            font-size: 0.72rem !important;
            letter-spacing: 1.2px !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: {ROYAL_GOLD} !important;
            font-family: 'Cinzel', serif !important;
            font-weight: 700 !important;
            font-size: 1.4rem !important;
        }}

        /* ============================================================
           PROGRESS BAR
        ============================================================ */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {ROYAL_GOLD} 0%, {EMBER_ORANGE} 100%) !important;
            border-radius: 4px !important;
        }}
        .stProgress > div > div {{
            background: {bg_secondary} !important;
            border-radius: 4px !important;
        }}

        /* ============================================================
           SLIDER
        ============================================================ */
        .stSlider [data-testid="stSliderThumb"] {{
            background: {ROYAL_GOLD} !important;
            border: 2px solid {bg_primary} !important;
            box-shadow: 0 0 0 2px {ROYAL_GOLD} !important;
        }}
        .stSlider [data-testid="stSliderTrack"] > div:first-child {{
            background: linear-gradient(90deg, {ROYAL_GOLD}, {EMBER_ORANGE}) !important;
        }}

        /* ============================================================
           CHECKBOX
        ============================================================ */
        .stCheckbox > label {{
            font-family: 'Outfit', sans-serif !important;
            color: {text_primary} !important;
            font-size: 0.9rem !important;
        }}

        /* ============================================================
           SELECTBOX
        ============================================================ */
        .stSelectbox > div > div {{
            background: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            transition: border-color 0.2s ease !important;
        }}
        .stSelectbox > div > div:focus-within {{
            border-color: {ROYAL_GOLD} !important;
        }}

        /* ============================================================
           DIVIDER
        ============================================================ */
        hr {{
            border: none !important;
            border-top: 1px solid {border_color} !important;
            margin: 16px 0 !important;
        }}

        /* ============================================================
           ORNAMENTAL DIVIDER
        ============================================================ */
        .ornament-divider {{
            text-align: center;
            color: {ROYAL_GOLD};
            font-size: 1rem;
            margin: 20px 0;
            opacity: 0.45;
            letter-spacing: 8px;
        }}

        /* ============================================================
           ROYAL CARD
        ============================================================ */
        .royal-card {{
            background: linear-gradient(135deg, {bg_card} 0%, {bg_secondary} 100%);
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 20px 22px;
            margin: 10px 0;
            box-shadow: 0 4px 16px {shadow_color};
            transition: all 0.25s ease;
        }}

        .royal-card:hover {{
            border-color: {border_hover};
            box-shadow: 0 8px 28px {shadow_hover};
            transform: translateY(-2px);
        }}

        /* ============================================================
           PILL / BADGE SYSTEM
        ============================================================ */
        .category-pill {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            letter-spacing: 0.5px;
            transition: all 0.2s ease;
        }}

        .pill-character {{ background: rgba(212, 175, 55, 0.12); color: {ROYAL_GOLD}; border: 1px solid rgba(212, 175, 55, 0.3); }}
        .pill-kingdom {{ background: rgba(45, 90, 61, 0.12); color: #5DAE7A; border: 1px solid rgba(45, 90, 61, 0.3); }}
        .pill-battle {{ background: rgba(139, 26, 26, 0.12); color: #D45B5B; border: 1px solid rgba(139, 26, 26, 0.3); }}
        .pill-timeline {{ background: rgba(100, 130, 180, 0.12); color: #7BA3D4; border: 1px solid rgba(100, 130, 180, 0.3); }}
        .pill-general {{ background: rgba(184, 169, 144, 0.12); color: #B8A990; border: 1px solid rgba(184, 169, 144, 0.3); }}
        .pill-inscope {{ background: rgba(45, 90, 61, 0.15); color: #5DAE7A; border: 1px solid rgba(45, 90, 61, 0.3); }}
        .pill-outscope {{ background: rgba(139, 26, 26, 0.15); color: #D45B5B; border: 1px solid rgba(139, 26, 26, 0.3); }}

        /* ============================================================
           PRESET QUERY PILLS
        ============================================================ */
        .preset-pill {{
            display: inline-block;
            padding: 7px 16px;
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 50px;
            font-family: 'Outfit', sans-serif;
            font-size: 0.82rem;
            color: {text_secondary};
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .preset-pill:hover {{
            border-color: {ROYAL_GOLD};
            color: {ROYAL_GOLD};
            background: {nav_hover_bg};
            transform: translateY(-1px);
        }}

        /* ============================================================
           STATUS INDICATORS
        ============================================================ */
        .status-dot-green {{
            display: inline-block;
            width: 8px; height: 8px;
            background: #5DAE7A;
            border-radius: 50%;
            margin-right: 6px;
            box-shadow: 0 0 6px rgba(93, 174, 122, 0.6);
            animation: pulse-green 2s ease-in-out infinite;
        }}
        @keyframes pulse-green {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}

        /* ============================================================
           SCROLLBAR
        ============================================================ */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {bg_primary};
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(212, 175, 55, 0.25);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(212, 175, 55, 0.45);
        }}

        /* ============================================================
           ALERT / TOAST
        ============================================================ */
        .stAlert {{
            border-radius: 12px !important;
            border-left: 4px solid {ROYAL_GOLD} !important;
            background: {bg_card} !important;
        }}

        /* ============================================================
           DATAFRAME / TABLE
        ============================================================ */
        .stDataFrame {{
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        /* ============================================================
           ANIMATIONS
        ============================================================ */
        @keyframes goldGlow {{
            0%   {{ text-shadow: 0 0 10px rgba(212, 175, 55, 0.15); }}
            50%  {{ text-shadow: 0 0 25px rgba(212, 175, 55, 0.45), 0 0 50px rgba(212, 175, 55, 0.1); }}
            100% {{ text-shadow: 0 0 10px rgba(212, 175, 55, 0.15); }}
        }}
        .glow-text {{
            animation: goldGlow 3s ease-in-out infinite;
        }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in-up {{
            animation: fadeInUp 0.4s ease forwards;
        }}

        @keyframes shimmer {{
            0%   {{ background-position: -200% center; }}
            100% {{ background-position: 200% center; }}
        }}

        /* ============================================================
           SECTION HEADER BAR
        ============================================================ */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 24px 0 12px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid {border_color};
        }}
        .section-header-title {{
            font-family: 'Cinzel', serif;
            font-size: 1.05rem;
            font-weight: 600;
            color: {ROYAL_GOLD};
            letter-spacing: 1px;
            margin: 0;
        }}
        .section-header-line {{
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, {border_color}, transparent);
        }}

        /* ============================================================
           HIDE STREAMLIT BRANDING (PRESERVE SIDEBAR TOGGLE)
        ============================================================ */
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

        /* ============================================================
           COLUMN RESULT CARDS
        ============================================================ */
        .result-card {{
            background: linear-gradient(160deg, {bg_card} 0%, {bg_secondary} 100%);
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 12px {shadow_color};
        }}
        .result-card-header {{
            font-family: 'Cinzel', serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: {ROYAL_GOLD};
            letter-spacing: 1px;
            border-bottom: 1px solid {border_color};
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}

        /* ============================================================
           SPINNER OVERRIDE
        ============================================================ */
        .stSpinner > div {{
            border-top-color: {ROYAL_GOLD} !important;
        }}
    </style>
    """


def render_ornament():
    """Return an ornamental divider HTML."""
    return '<div class="ornament-divider">─ ─ ─</div>'


# Material Symbols icon names for each category
_CATEGORY_ICONS = {
    "Character":  "person",
    "Kingdom":    "fort",
    "Battle":     "shield",
    "Timeline":   "calendar_today",
    "General":    "history_edu",
}


def _msi(name: str) -> str:
    """Inline Material Symbol icon span using Streamlit's native class."""
    return f'<span class="material-symbols-rounded msi">{name}</span>'


def render_category_pill(category: str) -> str:
    """Render a styled category pill badge."""
    css_class = f"pill-{category.lower()}"
    icon_name = _CATEGORY_ICONS.get(category, "help")
    return f'<span class="category-pill {css_class}">{_msi(icon_name)}{category}</span>'


def render_scope_pill(is_in_scope: bool) -> str:
    """Render an in-scope/out-of-scope pill badge."""
    if is_in_scope:
        return f'<span class="category-pill pill-inscope">{_msi("check_circle")}In-Scope</span>'
    else:
        return f'<span class="category-pill pill-outscope">{_msi("cancel")}Out-of-Scope</span>'


def render_section_header(title: str, icon: str = "") -> str:
    """Render a styled section header with decorative line.
    icon: a Material Symbols ligature name (e.g. 'search', 'analytics').
    """
    icon_html = f'<span class="material-symbols-rounded msi" style="font-size:1.1em; color:inherit;">{icon}</span> ' if icon else ""
    return f'''
    <div class="section-header">
        <span class="section-header-title">{icon_html}{title}</span>
        <div class="section-header-line"></div>
    </div>
    '''
