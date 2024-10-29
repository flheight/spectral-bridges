import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define dimensions and scores for MNIST
mnist_dim = ['h=8', 'h=16', 'h=32', 'h=64', 'h=784 (full)']
mnist_km_ari = [0.3625, 0.3661, 0.3726, 0.3821, 0.3665]
mnist_em_ari = [0.4587, 0.4974, 0.4799, 0.4440, 0.1850]
mnist_wc_ari = [0.4240, 0.4716, 0.5068, 0.4904, 0.4297]
mnist_sb_ari = [0.5985, 0.6876, 0.7213, 0.7069, 0.6619]

mnist_km_nmi = [0.4750, 0.4890, 0.4957, 0.5036, 0.4915]
mnist_em_nmi = [0.5921, 0.6424, 0.6318, 0.5958, 0.3252]
mnist_wc_nmi = [0.5467, 0.6220, 0.6577, 0.6536, 0.6129]
mnist_sb_nmi = [0.6709, 0.7627, 0.7960, 0.7854, 0.7628]

# Define dimensions and scores for FMNIST
fmnist_dim = ['h=8', 'h=16', 'h=32', 'h=64', 'h=784 (full)']
fmnist_km_ari = [0.3730, 0.3960, 0.3796, 0.3877, 0.3497]
fmnist_em_ari = [0.3960, 0.3900, 0.4089, 0.4426, 0.3247]
fmnist_wc_ari = [0.3796, 0.4089, 0.3292, 0.3619, 0.3684]
fmnist_sb_ari = [0.3877, 0.4426, 0.4488, 0.4489, 0.4088]

fmnist_km_nmi = [0.5191, 0.5739, 0.5233, 0.5715, 0.5055]
fmnist_em_nmi = [0.5739, 0.5843, 0.5945, 0.6232, 0.4889]
fmnist_wc_nmi = [0.5233, 0.5945, 0.5231, 0.5511, 0.5715]
fmnist_sb_nmi = [0.5715, 0.6232, 0.6321, 0.6268, 0.6122]

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
# Adjust vertical spacing to double the gap between subplots
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.3)

# Methods and their corresponding scores for MNIST
methods = ['KM', 'EM', 'WC', 'SB']
mnist_ari_scores = [mnist_km_ari, mnist_em_ari, mnist_wc_ari, mnist_sb_ari]
mnist_nmi_scores = [mnist_km_nmi, mnist_em_nmi, mnist_wc_nmi, mnist_sb_nmi]

# Plot bars for MNIST ARI scores
for i, (method, score) in enumerate(zip(methods, mnist_ari_scores)):
    max_scores = [max(mnist_km_ari[i], mnist_em_ari[i], mnist_wc_ari[i], mnist_sb_ari[i]) for i in range(len(mnist_dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=mnist_dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(mnist_dim))],
        width=0.2,
        showlegend=True
    ), row=1, col=1)

# Plot bars for MNIST NMI scores
for i, (method, score) in enumerate(zip(methods, mnist_nmi_scores)):
    max_scores = [max(mnist_km_nmi[i], mnist_em_nmi[i], mnist_wc_nmi[i], mnist_sb_nmi[i]) for i in range(len(mnist_dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=mnist_dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(mnist_dim))],
        width=0.2,
        showlegend=False
    ), row=1, col=2)

# Plot bars for FMNIST ARI scores
for i, (method, score) in enumerate(zip(methods, [fmnist_km_ari, fmnist_em_ari, fmnist_wc_ari, fmnist_sb_ari])):
    max_scores = [max(fmnist_km_ari[i], fmnist_em_ari[i], fmnist_wc_ari[i], fmnist_sb_ari[i]) for i in range(len(fmnist_dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=fmnist_dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(fmnist_dim))],
        width=0.2,
        showlegend=False
    ), row=2, col=1)

# Plot bars for FMNIST NMI scores
for i, (method, score) in enumerate(zip(methods, [fmnist_km_nmi, fmnist_em_nmi, fmnist_wc_nmi, fmnist_sb_nmi])):
    max_scores = [max(fmnist_km_nmi[i], fmnist_em_nmi[i], fmnist_wc_nmi[i], fmnist_sb_nmi[i]) for i in range(len(fmnist_dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=fmnist_dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(fmnist_dim))],
        width=0.2,
        showlegend=False
    ), row=2, col=2)

# Add centered titles for MNIST and FMNIST at a fixed height above their respective subplots
mnist_title_y = 1.1  # Fixed height for MNIST title above its subplot
fmnist_title_y = 0.4  # Fixed height for FMNIST title above its subplot
fig.add_annotation(text="MNIST", x=0.5, y=mnist_title_y, showarrow=False, font=dict(size=28, family="Arial"), xref="paper", yref="paper")
fig.add_annotation(text="FMNIST", x=0.5, y=fmnist_title_y, showarrow=False, font=dict(size=28, family="Arial"), xref="paper", yref="paper")

# Update layout
fig.update_layout(
    plot_bgcolor='whitesmoke',
    width=1200,
    height=800,
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
fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='ARI Score', row=1, col=1, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='NMI Score', row=1, col=2, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='ARI Score', row=2, col=1, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='NMI Score', row=2, col=2, title_font=dict(size=28), tickfont=dict(size=24))

# Save the figure as a PDF file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, "mnist_fmnist_summary.pdf")
