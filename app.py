import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np

# Generate sample ECG data
def generate_ecg_data(length=1000):
    t = np.linspace(0, 10, length)
    # Simulate ECG waveform with multiple sine waves and noise
    ecg = (np.sin(2 * np.pi * t) +
           0.5 * np.sin(4 * np.pi * t) +
           0.3 * np.sin(6 * np.pi * t) +
           0.1 * np.random.normal(0, 0.05, length))
    return t, ecg

# Create 3D data with spiral effect
def create_3d_data(t, ecg, scale=1.0):
    x = t * 10  # Stretch time for better visualization
    y = ecg * scale
    z = 5 * np.sin(t * 2)  # Spiral in z-direction
    return x, y, z

# Create animated frames for wow effect
def create_frames(num_frames=50):
    frames = []
    t, ecg = generate_ecg_data()
    for i in range(num_frames):
        # Pulse effect: scale amplitude to simulate heartbeat
        pulse_scale = 1 + 0.3 * np.sin(2 * np.pi * i / num_frames)
        x, y, z = create_3d_data(t, ecg, pulse_scale)
        # Color gradient based on amplitude
        colors = np.abs(y) * 255 / np.max(np.abs(y))
        frame = go.Frame(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                line=dict(color='rgba(255, 0, 100, 0.8)', width=3),
                marker=dict(size=2, color=colors, colorscale='Plasma', showscale=False)
            )]
        )
        frames.append(frame)
    return frames

# Initial figure
t, ecg = generate_ecg_data()
x, y, z = create_3d_data(t, ecg)
colors = np.abs(y) * 255 / np.max(np.abs(y))

fig = go.Figure(
    data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color='rgba(255, 0, 100, 0.8)', width=3),
        marker=dict(size=2, color=colors, colorscale='Plasma', showscale=False)
    )],
    frames=create_frames()
)

# Add animation settings
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Time (s)', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.3)'),
        yaxis=dict(title='Amplitude', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.3)'),
        zaxis=dict(title='Z-Dimension', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.3)'),
        bgcolor='rgba(10, 10, 30, 0.9)'  # Dark space-like background
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title=dict(text="3D ECG Visualization: Pulse of Life", font=dict(color='white', size=24)),
    updatemenus=[dict(
        type='buttons',
        buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), mode='immediate')]),
                 dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])]
    )]
)

# Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'backgroundColor': '#0a0a1e', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("🫀 3D ECG Website: Creative Wow Effects", style={'textAlign': 'center', 'color': '#00f0ff'}),
    html.P("Experience the heartbeat in stunning 3D! Rotate, zoom, and watch the pulse animation.", style={'textAlign': 'center'}),
    dcc.Graph(
        id='3d-ecg-graph',
        figure=fig,
        style={'height': '80vh'}
    ),
    html.Div([
        html.P("💡 Tip: Use mouse to rotate the view and discover the ECG waveform in 3D space!", style={'textAlign': 'center', 'marginTop': '20px'})
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)