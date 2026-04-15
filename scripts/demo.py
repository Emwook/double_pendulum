import torch as tr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from pinn_network import PINN_Network
from get_data import get_data

def run_standalone_demo(model_path="results/pinn_pendulum_v2.pth", trial_to_test=3):
    stride = 10
    dt = 0.001 * stride
    L1, L2 = 0.2, 0.2
    seq_len = 200 
    
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        return

    model = PINN_Network(input_size=2, hidden_size=32).to(device)
    model.load_state_dict(tr.load(model_path, map_location=device))
    model.eval()

    data = get_data(trial_num=trial_to_test, stride=stride).to(device)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    line_act, = ax.plot([], [], 'o-', lw=2, color='#333333', label='Actual Tracking')
    line_pred, = ax.plot([], [], 'o--', lw=2, color='#FF3333', label='PINN Prediction', alpha=0.8)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    ax.legend(loc='lower right')
    ax.set_title("Double pendulum: Video tracking (source) vs. PINN GRU model", fontsize=12)

    def get_coords(th):
        x1 = L1 * np.sin(th[0])
        y1 = -L1 * np.cos(th[0])
        x2 = x1 + L2 * np.sin(th[1])
        y2 = y1 - L2 * np.cos(th[1])
        return [0, x1, x2], [0, y1, y2]

    def update(i):
        idx = i + seq_len
        if idx >= len(data): return line_act, line_pred, time_text
        
        x_window = data[i : idx].unsqueeze(0)
        y_actual = data[idx].cpu().numpy()
        
        with tr.no_grad():
            y_pred = model(x_window).squeeze().cpu().numpy()

        act_x, act_y = get_coords(y_actual)
        pred_x, pred_y = get_coords(y_pred)

        line_act.set_data(act_x, act_y)
        line_pred.set_data(pred_x, pred_y)
        time_text.set_text(f'Time: {idx*dt:.2f}s')
        
        return line_act, line_pred, time_text

    frames_to_play = len(data) - seq_len
    ani = FuncAnimation(fig, update, frames=range(0, frames_to_play, 2), 
                        interval=30, blit=True, repeat=True)
    
    # ani.save('results/pendulum_comparison_3.gif', writer='ffmpeg', fps=20)
    plt.show()

run_standalone_demo()