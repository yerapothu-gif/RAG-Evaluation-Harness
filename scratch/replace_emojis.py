import os
import re

EMOJI_MAP = {
    "⚜️": ":material/star:",
    "⚜": ":material/star:",
    "👤": ":material/person:",
    "🏰": ":material/fort:",
    "⚔️": ":material/shield:",
    "⚔": ":material/shield:",
    "📅": ":material/calendar_today:",
    "📜": ":material/history_edu:",
    "✅": ":material/check_circle:",
    "❌": ":material/cancel:",
    "❓": ":material/help:",
    "🔍": ":material/search:",
    "⚙️": ":material/settings:",
    "⚙": ":material/settings:",
    "🏠": ":material/home:",
    "🌙": ":material/dark_mode:",
    "☀️": ":material/light_mode:",
    "📊": ":material/bar_chart:",
    "📈": ":material/show_chart:",
    "🕸️": ":material/hub:",
    "🕸": ":material/hub:",
    "🔑": ":material/key:",
    "🟡": ":material/radio_button_checked:",
    "🟠": ":material/radio_button_checked:",
    "⏱️": ":material/timer:",
    "⏱": ":material/timer:",
    "📄": ":material/description:",
    "🚀": ":material/rocket_launch:",
    "🔄": ":material/sync:",
    "📭": ":material/inbox:",
    "✍️": ":material/edit:",
    "✍": ":material/edit:",
    "🧠": ":material/psychology:",
    "ℹ️": ":material/info:",
    "ℹ": ":material/info:",
    "⏳": ":material/hourglass_empty:",
    "⚠️": ":material/warning:",
    "🎯": ":material/target:",
    "👑": ":material/crown:" # if target/crown exist, otherwise they just won't render
}

def replace_in_file(filepath):
    if not os.path.exists(filepath):
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    original = content
    for emoji, icon in EMOJI_MAP.items():
        content = content.replace(emoji, icon)
        
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes in {filepath}")

def main():
    files = [
        "app.py",
        "src/theme.py",
        "src/query_classifier.py",
        "src/guardrail.py"
    ]
    for f in files:
        replace_in_file(f)

if __name__ == "__main__":
    main()
