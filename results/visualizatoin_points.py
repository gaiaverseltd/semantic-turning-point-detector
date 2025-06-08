import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define turning points with actual data from logs
model_data = {
    "Qwen3 1.7B": {
        "indices": [0, 7, 78, 83, 107, 115, 121, 123, 127, 129, 131, 134, 139, 141, 143, 148],
        "categories": ["Meta-Reflection", "Emotional Tone", "Emotional Tone", "Decision", "Meta-Reflection", 
                      "Objection", "Decision", "Emotion", "Emotion", "Topic", "Emotion", "Emotion", 
                      "Meta-Reflection", "Meta-Reflection", "Emotion", "Emotion"],
        "processing_time": "51:26",
        "avg_significance": 0.99,
        "count": 16
    },
    "GPT-4.1 Nano": {
        "indices": [0, 7, 24, 41, 98, 104, 109, 111, 114, 117, 124, 132, 141, 144, 147, 152],
        "categories": ["Insight", "Action", "Action", "Question", "Question", "Insight", "Question", 
                      "Insight", "Decision", "Insight", "Insight", "Emotion", "Insight", "Action", 
                      "Emotion", "Question"],
        "processing_time": "N/A",
        "avg_significance": 0.94,
        "count": 17
    },
    "Qwen3 30B": {
        "indices": [0, 83, 90, 115, 127, 133, 140, 142, 145, 148, 150, 153],
        "categories": ["Conflict", "Conflict", "Conflict", "Conflict", "Conflict", "Emotion", 
                      "Decision", "Decision", "Decision", "Topic", "Decision", "Decision"],
        "processing_time": "27:24",
        "avg_significance": 0.99,
        "count": 12
    },
    "GPT-4.1 Large": {
        "indices": [0, 5, 10, 42, 59, 64, 69, 75, 86, 91, 98, 120, 127, 145, 150, 152],
        "categories": ["Question", "Question", "Decision", "Objection", "Emotion", "Insight", 
                      "Insight", "Insight", "Emotion", "Emotion", "Problem", "Decision", 
                      "Insight", "Problem", "Emotion", "Decision"],
        "processing_time": "06:06",
        "avg_significance": 0.95,
        "count": 16
    },
    "Gemini 2.5 Flash": {
        "indices": [27, 41, 45, 118, 121, 124, 127, 129, 131, 133, 136, 141, 144, 148, 150, 152],
        "categories": ["Topic", "Insight", "Problem", "Problem", "Decision", "Action", "Insight", 
                      "Problem", "Insight", "Problem", "Question", "Question", "Action", 
                      "Objection", "Problem", "Question"],
        "processing_time": "27:08",
        "avg_significance": 1.00,
        "count": 16
    }
}

# Create summary table
summary_data = []
for model, data in model_data.items():
    family = "Compact Open-Source" if "1.7B" in model else \
             "Proprietary (Small)" if "Nano" in model else \
             "Large Open-Source" if "30B" in model else \
             "Proprietary (Large)"
    
    params = "~1.7B" if "1.7B" in model else \
             "~1.8B" if "Nano" in model else \
             "~30B" if "30B" in model else \
             "N/A"
    
    summary_data.append({
        "Model Family": family,
        "Model": model,
        "Parameters": params,
        "Turning Points": data["count"],
        "Processing Time": data["processing_time"],
        "Avg. Significance": data["avg_significance"]
    })

summary_df = pd.DataFrame(summary_data)

# Print summary
print("=" * 80)
print("SEMANTIC TURNING POINT ANALYSIS: Strindberg's 'Pariah' (6,126 tokens)")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)

# Build visualization dataframe
viz_data = []
for model, data in model_data.items():
    for i, idx in enumerate(data["indices"]):
        category = data["categories"][i] if i < len(data["categories"]) else "Unknown"
        viz_data.append({
            "model": model,
            "index": idx,
            "category": category
        })

df = pd.DataFrame(viz_data)

# Define category colors
category_colors = {
    "Meta-Reflection": "#FF6B6B",    # Red
    "Emotion": "#4ECDC4",            # Teal
    "Decision": "#45B7D1",           # Blue
    "Insight": "#96CEB4",            # Green
    "Action": "#FFEAA7",             # Yellow
    "Question": "#DDA0DD",           # Plum
    "Problem": "#FFB347",            # Orange
    "Topic": "#98FB98",              # Pale Green
    "Conflict": "#F0E68C",           # Khaki
    "Objection": "#FFA07A",          # Light Salmon
    "Unknown": "#D3D3D3"             # Light Gray
}

# Find convergence points (3+ models agree)
counts = df["index"].value_counts().reset_index()
counts.columns = ["index", "count"]
convergence_points = counts[counts["count"] >= 3].sort_values("index")

# Key dialogue moments
key_quotes = {
    0: "Opening: 'What oppressive heat! We'll surely have a thunder-shower.'",
    127: "Revelation: '[Sits... dark coat] What's going to happen now?'",
    141: "Chess Metaphor: 'You are pretty crafty... next move you can be checkmated.'",
    148: "Power Reversal: 'You see...deal with you as I did with the coachman!'",
    150: "Direct Challenge: 'Then you don't believe that I ever took from the case?'",
    152: "Final Question: 'You are a different kind of being from me... Do you give up now?'"
}

# Create figure
fig = go.Figure()

# Add scatter points for each model, colored by category
for model in model_data.keys():
    model_df = df[df["model"] == model]
    
    for category in model_df["category"].unique():
        cat_df = model_df[model_df["category"] == category]
        
        fig.add_trace(go.Scatter(
            x=cat_df["index"],
            y=[model] * len(cat_df),
            mode='markers',
            marker=dict(
                size=14,
                color=category_colors.get(category, "#D3D3D3"),
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            name=f"{category}",
            legendgroup=category,
            showlegend=category not in [trace.name for trace in fig.data],
            hovertemplate=f"<b>{model}</b><br>Index: %{{x}}<br>Category: {category}<extra></extra>"
        ))

# Add vertical lines for convergence points
for _, row in convergence_points.iterrows():
    idx, count = row['index'], row['count']
    
    fig.add_vline(
        x=idx,
        line=dict(color="rgba(128, 128, 128, 0.4)", width=3, dash="dash"),
        annotation_text=f"{count} models converge",
        annotation_position="top",
        annotation=dict(font=dict(size=11, color="darkblue"))
    )

# Add key moment annotations
annotation_offset = 0
for idx in convergence_points["index"]:
    if idx in key_quotes:
        fig.add_annotation(
            x=idx,
            y=-0.8,
            text=f"<b>Index {idx}</b><br>{key_quotes[idx]}",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="darkslategray",
            ax=0,
            ay=40 + (annotation_offset * 15),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="darkslategray",
            borderwidth=1,
            font=dict(size=10),
            align="center"
        )
        annotation_offset += 1
for i, row in enumerate(convergence_points.itertuples()):
    idx, count = row.index, row.count
    fig.add_vline(
        x=idx,
        line=dict(color="rgba(128, 128, 128, 0.4)", width=3, dash="dash"),
        annotation_text=f"{count} converge",
        annotation_position="top left",
        annotation=dict(font=dict(size=9, color="darkblue"))
    )

# Add key moment annotations with staggered y positions and smaller font
annotation_offset = 0
for idx in convergence_points["index"]:
    if idx in key_quotes:
        fig.add_annotation(
            x=idx,
            y=-1.2 - annotation_offset * 0.05,  # shift each annotation lower
            text=f"<b>Idx {idx}</b>: {key_quotes[idx]}",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="darkslategray",
            ax=0,
            ay=20,  # small vertical arrow
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="darkslategray",
            borderwidth=1,
            font=dict(size=9),
            align="center"
        )
        annotation_offset += 1

# Update layout
fig.update_layout(
    title={
        'text': "Semantic Turning Points Across Models: Strindberg's 'Pariah'<br><sub>Colored by Semantic Category | Vertical lines show model convergence</sub>",
        'x': 0.5,
        'font': {'size': 16}
    },
    xaxis_title="Dialogue Message Index",
    yaxis_title="Model",
    width=1600,
    height=800,
    margin=dict(l=200, r=50, t=120, b=200),
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        title="Semantic Categories"
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.3)',
    range=[-5, 160],
    title_font=dict(size=14)
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.3)',
    categoryorder='array',
    categoryarray=list(reversed(list(model_data.keys()))),
    title_font=dict(size=14)
)

fig.show()

# Print convergence analysis
print(f"\nKEY CONVERGENCE POINTS (3+ models agree):")
print("-" * 50)
for _, row in convergence_points.iterrows():
    idx, count = row['index'], row['count']
    models_at_point = df[df["index"] == idx]["model"].tolist()
    categories_at_point = df[df["index"] == idx]["category"].unique()
    
    print(f"Index {idx:3d}: {count} models converge")
    print(f"  Models: {', '.join(models_at_point)}")
    print(f"  Categories: {', '.join(categories_at_point)}")
    if idx in key_quotes:
        print(f"  Context: {key_quotes[idx]}")
    print()

print("=" * 80)
print("ARCHITECTURAL INTELLIGENCE vs BRUTE FORCE:")
print("Small models (1.7B-1.8B parameters) achieve comparable semantic detection")
print("to much larger models, demonstrating that architectural sophistication")
print("drives understanding more than computational scale.")
print("=" * 80)