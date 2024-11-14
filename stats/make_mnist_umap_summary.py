import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define the dimensions and scores
dim = ['h=2', 'h=4', 'h=8', 'h=16']

# Adjusted Rand Index (ARI) Scores
km_ari = [0.7843, 0.7853, 0.7648, 0.7795]
em_ari = [0.8009, 0.8368, 0.8565, 0.8469]
wc_ari = [0.7460, 0.7674, 0.7757, 0.7986]
git_ari = [0.9316, 0.9269]
sb_ari = [0.8992, 0.9273, 0.9296, 0.9318]

# Normalized Mutual Information (NMI) Scores
km_nmi = [0.8471, 0.8524, 0.8374, 0.8471]
em_nmi = [0.8615, 0.8791, 0.8915, 0.8831]
wc_nmi = [0.8403, 0.8518, 0.8514, 0.8710]
git_nmi = [0.9195, 0.9148]
sb_nmi = [0.9023, 0.9159, 0.9169,0.9202]

# Define colors for each method and their lighter shades
colors = {
    'KM': ('blue', 'lightblue'),
    'EM': ('green', 'lightgreen'),
    'WC': ('red', 'lightcoral'),
    'SB': ('purple', 'plum')
}

# Helper function to determine the color of each bar
def get_bar_color(score, max_score, method):
    return colors[method][0] if score == max_score else colors[method][1]

# Create the figure with subplots for ARI and NMI
fig = make_subplots(rows=1, cols=2)

# Methods and their corresponding scores
methods = ['KM', 'EM', 'WC', 'SB']
ari_scores = [km_ari, em_ari, wc_ari, sb_ari]
nmi_scores = [km_nmi, em_nmi, wc_nmi, sb_nmi]

# Plot bars for each method for ARI scores
for i, (method, score) in enumerate(zip(methods, ari_scores)):
    max_scores = [max(km_ari[i], em_ari[i], wc_ari[i], sb_ari[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=True  # Only show legend once
    ), row=1, col=1)

# Plot bars for each method for NMI scores
for i, (method, score) in enumerate(zip(methods, nmi_scores)):
    max_scores = [max(km_nmi[i], em_nmi[i], wc_nmi[i], sb_nmi[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=False  # Only show legend once
    ), row=1, col=2)

# Update layout with larger and bold axis titles and tick labels
fig.update_layout(
    title={
        'text': "MNIST - UMAP",
        'x': 0.5,  # Center the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 36,  # Make the title font larger
        }
    },
    plot_bgcolor='whitesmoke',
    width=1200,
    height=600,
    xaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    yaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    barmode='group',
    legend=dict(
        font=dict(
            size=20,
            weight='bold'
        ),
        title_font_size=24,
        itemsizing='constant'
    )
)

fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=22))
fig.update_yaxes(title_text='ARI Score', row=1, col=1, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='NMI Score', row=1, col=2, title_font=dict(size=28), tickfont=dict(size=24))

# Save the figure as a pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, "mnist_umap_summary.pdf")
