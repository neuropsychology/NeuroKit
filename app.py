import dash
from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Generate sample ECG data
def generate_ecg_data(num_points=1000, heart_rate=60):
    t = np.linspace(0, 10, num_points)  # 10 seconds
    # Simulate ECG waveform with P, QRS, T waves
    ecg = (
        0.5 * np.sin(2 * np.pi * heart_rate / 60 * t) +  # Base rhythm
        0.3 * np.exp(-((t % 1) - 0.1)**2 / 0.01) +  # P wave
        -1.0 * np.exp(-((t % 1) - 0.2)**2 / 0.005) +  # Q wave
        1.5 * np.exp(-((t % 1) - 0.25)**2 / 0.005) +  # R wave
        -0.5 * np.exp(-((t % 1) - 0.3)**2 / 0.01) +  # S wave
        0.4 * np.exp(-((t % 1) - 0.4)**2 / 0.02)  # T wave
    )
    # Add noise
    ecg += 0.1 * np.random.normal(0, 1, num_points)
    return t, ecg

# Create 3D visualization
def create_3d_ecg_plot(t, ecg):
    # Create a spiral effect for wow factor
    theta = 2 * np.pi * t / max(t) * 5  # 5 spirals
    x = t
    y = ecg
    z = np.sin(theta) * 0.5 + np.cos(theta) * 0.5

    # Color based on amplitude
    colors = np.abs(ecg)
    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    color_scale = [[0, 'blue'], [0.5, 'red'], [1, 'yellow']]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        line=dict(width=4, color=colors, colorscale=color_scale),
        name='ECG Signal'
    )])

    fig.update_layout(
        title='3D ECG Visualization - Pulse of Life',
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Amplitude (mV)',
            zaxis_title='Spiral Dimension',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white'
    )

    # Add animation frames for beating effect
    frames = []
    for i in range(0, len(t), 10):
        frame_data = go.Scatter3d(
            x=x[:i+1],
            y=y[:i+1],
            z=z[:i+1],
            mode='lines',
            line=dict(width=4, color=colors[:i+1], colorscale=color_scale)
        )
        frames.append(go.Frame(data=[frame_data]))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=False), fromcurrent=True, mode='immediate')])]
        )]
    )

    return fig

# Dash app
app = dash.Dash(__name__)

t, ecg = generate_ecg_data()
fig = create_3d_ecg_plot(t, ecg)

app.layout = html.Div([
    html.H1('3D ECG Website - Creative Wow Effects', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': 'black'}),
    html.P('Experience the heart\'s rhythm in stunning 3D! Rotate, zoom, and watch the pulse come alive.', style={'textAlign': 'center', 'color': 'white'}),
    dcc.Graph(figure=fig, style={'height': '80vh'})
], style={'backgroundColor': 'black', 'height': '100vh'})

if __name__ == '__main__':
    app.run_server(debug=True)