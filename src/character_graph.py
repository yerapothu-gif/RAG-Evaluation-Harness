"""
3D Character Relationship Graph using NetworkX + Plotly.
Visualizes character relationships from the Baahubali saga.
"""
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from src.config import ROYAL_GOLD, DEEP_MAROON, DARK_BROWN, BLOOD_RED, WARM_CREAM


# Character data with roles and descriptions
CHARACTERS = {
    "Amarendra\nBaahubali": {
        "role": "protagonist",
        "description": "Rightful heir to Mahishmati. Brave, compassionate warrior-king.",
        "group": 1,
    },
    "Mahendra\nBaahubali": {
        "role": "protagonist",
        "description": "Son of Amarendra. Grew up as Shivudu. Reclaims the throne.",
        "group": 1,
    },
    "Sivagami": {
        "role": "neutral",
        "description": "Rajmata of Mahishmati. Powerful regent. Mother of Bhallaladeva.",
        "group": 2,
    },
    "Devasena": {
        "role": "protagonist",
        "description": "Princess of Kuntala. Fierce warrior-archer. Wife of Amarendra.",
        "group": 1,
    },
    "Bhallaladeva": {
        "role": "antagonist",
        "description": "Tyrant king. Consumed by jealousy. Rival of Amarendra.",
        "group": 3,
    },
    "Kattappa": {
        "role": "neutral",
        "description": "Loyal commander. Bound by oath. Killed Amarendra under orders.",
        "group": 2,
    },
    "Bijjaladeva": {
        "role": "antagonist",
        "description": "Father of Bhallaladeva. Bitter manipulator. Puppet master.",
        "group": 3,
    },
    "Avantika": {
        "role": "protagonist",
        "description": "Rebel warrior. Love interest of Mahendra Baahubali.",
        "group": 1,
    },
    "Kumara\nVarma": {
        "role": "protagonist",
        "description": "Devasena's nephew. Prince of Kuntala. Ally.",
        "group": 1,
    },
    "Kalakeya": {
        "role": "antagonist",
        "description": "Enemy chieftain who invaded Mahishmati.",
        "group": 3,
    },
}

# Relationships (source, target, relationship, strength)
RELATIONSHIPS = [
    ("Amarendra\nBaahubali", "Mahendra\nBaahubali", "Father", 3),
    ("Devasena", "Mahendra\nBaahubali", "Mother", 3),
    ("Amarendra\nBaahubali", "Devasena", "Husband", 3),
    ("Sivagami", "Amarendra\nBaahubali", "Adoptive\nMother", 2),
    ("Sivagami", "Bhallaladeva", "Mother", 3),
    ("Bijjaladeva", "Bhallaladeva", "Father", 3),
    ("Bijjaladeva", "Sivagami", "Husband", 2),
    ("Bhallaladeva", "Amarendra\nBaahubali", "Rival", 3),
    ("Kattappa", "Amarendra\nBaahubali", "Mentor &\nBetrayer", 3),
    ("Kattappa", "Mahendra\nBaahubali", "Ally", 2),
    ("Avantika", "Mahendra\nBaahubali", "Lover", 2),
    ("Kumara\nVarma", "Devasena", "Nephew", 1),
    ("Kalakeya", "Amarendra\nBaahubali", "Enemy", 2),
    ("Bijjaladeva", "Amarendra\nBaahubali", "Schemer\nAgainst", 2),
]


def create_3d_graph() -> go.Figure:
    """Create a 3D interactive character relationship graph."""
    G = nx.Graph()

    # Add nodes
    for name, data in CHARACTERS.items():
        G.add_node(name, **data)

    # Add edges
    for source, target, relationship, strength in RELATIONSHIPS:
        G.add_edge(source, target, relationship=relationship, weight=strength)

    # 3D spring layout
    np.random.seed(42)
    pos = nx.spring_layout(G, dim=3, k=2.5, iterations=100, seed=42)

    # Separate coordinates
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]

    # Node colors based on role
    role_colors = {
        "protagonist": "#D4AF37",    # Royal Gold
        "antagonist": "#8B1A1A",      # Blood Red
        "neutral": "#7B8794",         # Silver/Steel
    }

    node_colors = [role_colors.get(CHARACTERS[node]["role"], "#7B8794") for node in G.nodes()]
    node_sizes = [28 if CHARACTERS[node]["role"] == "protagonist" else 22 for node in G.nodes()]

    # Hover text
    hover_texts = []
    for node in G.nodes():
        data = CHARACTERS[node]
        connections = list(G.neighbors(node))
        conn_str = ", ".join([c.replace("\n", " ") for c in connections[:5]])
        hover_texts.append(
            f"<b>{node.replace(chr(10), ' ')}</b><br>"
            f"Role: {data['role'].title()}<br>"
            f"{data['description']}<br>"
            f"Connected to: {conn_str}"
        )

    # Create edge traces
    edge_traces = []
    edge_label_x, edge_label_y, edge_label_z, edge_labels = [], [], [], []

    for source, target, relationship, strength in RELATIONSHIPS:
        x0, y0, z0 = pos[source]
        x1, y1, z1 = pos[target]

        # Edge line
        width = strength * 1.2
        # Color edges by relationship type
        if relationship in ["Rival", "Enemy", "Schemer\nAgainst"]:
            edge_color = "rgba(139, 26, 26, 0.6)"  # Red for conflict
        elif relationship in ["Mentor &\nBetrayer"]:
            edge_color = "rgba(180, 120, 50, 0.7)"  # Amber for complex
        elif relationship in ["Father", "Mother", "Son", "Husband"]:
            edge_color = "rgba(212, 175, 55, 0.5)"  # Gold for family
        else:
            edge_color = "rgba(123, 135, 148, 0.4)"  # Silver for others

        edge_traces.append(
            go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode="lines",
                line=dict(color=edge_color, width=width),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Edge label position (midpoint)
        edge_label_x.append((x0 + x1) / 2)
        edge_label_y.append((y0 + y1) / 2)
        edge_label_z.append((z0 + z1) / 2)
        edge_labels.append(relationship.replace("\n", " "))

    # Edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x,
        y=edge_label_y,
        z=edge_label_z,
        mode="text",
        text=edge_labels,
        textfont=dict(size=8, color="rgba(232, 220, 200, 0.7)", family="Arial"),
        hoverinfo="none",
        showlegend=False,
    )

    # Node trace
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.95,
            line=dict(color="rgba(232, 220, 200, 0.4)", width=1.5),
            symbol="circle",
        ),
        text=[n.replace("\n", " ") for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color=WARM_CREAM, family="Arial Black"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    # Glow effect (larger transparent markers behind nodes)
    glow_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(
            size=[s * 1.8 for s in node_sizes],
            color=node_colors,
            opacity=0.15,
            line=dict(width=0),
        ),
        hoverinfo="none",
        showlegend=False,
    )

    # Assemble figure
    fig = go.Figure(data=[glow_trace] + edge_traces + [edge_label_trace, node_trace])

    fig.update_layout(
        title=dict(
            text="⚔️ Baahubali Character Relationships",
            font=dict(size=20, color=ROYAL_GOLD, family="Georgia, serif"),
            x=0.5,
        ),
        scene=dict(
            bgcolor="rgba(14, 10, 7, 1)",
            xaxis=dict(
                showgrid=False, showticklabels=False, zeroline=False,
                title="", showspikes=False, visible=False,
            ),
            yaxis=dict(
                showgrid=False, showticklabels=False, zeroline=False,
                title="", showspikes=False, visible=False,
            ),
            zaxis=dict(
                showgrid=False, showticklabels=False, zeroline=False,
                title="", showspikes=False, visible=False,
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="rgba(14, 10, 7, 1)",
        plot_bgcolor="rgba(14, 10, 7, 1)",
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
        # Legend for role colors
        annotations=[
            dict(
                text="🟡 Protagonist  🔴 Antagonist  ⚪ Neutral",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                font=dict(size=12, color=WARM_CREAM),
            )
        ],
    )

    return fig
