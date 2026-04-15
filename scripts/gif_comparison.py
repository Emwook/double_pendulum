import torch as tr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pinn_network import PINN_Network
from get_data import get_data

def animate_comparison(model_path="results/pinn_pendulum_v2.pth", trial_num=1):
    stride = 10
    dt = 0.001 * stride
    L1, L2 = 0.2, 0.2
    
    model = PINN_Network(2, 32)
    model.load_state_dict(tr.load(model_path))
    model.eval()
    
    data = get_data(trial_num, stride=stride)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    line_act, = ax.plot([], [], 'o-', lw=2, color='black', label='source data')
    line_pred, = ax.plot([], [], 'o--', lw=2, color='red', label='model prediction', alpha=0.7)
    
    ax.legend()

    def update(i):
        if i < 200: return line_act, line_pred
        
        x_window = data[i-200 : i].unsqueeze(0)
        y_actual = data[i].numpy()
        
        with tr.no_grad():
            y_pred = model(x_window).squeeze().numpy()

        def get_coords(th):
            x1 = L1 * np.sin(th[0])
            y1 = -L1 * np.cos(th[0])
            x2 = x1 + L2 * np.sin(th[1])
            y2 = y1 - L2 * np.cos(th[1])
            return [0, x1, x2], [0, y1, y2]

        act_x, act_y = get_coords(y_actual)
        pred_x, pred_y = get_coords(y_pred)

        line_act.set_data(act_x, act_y)
        line_pred.set_data(pred_x, pred_y)
        
        return line_act, line_pred

    ani = FuncAnimation(fig, update, frames=range(200, len(data), 2), 
                        interval=20, blit=True)
    
    ani.save('results/pendulum_comparison_2.gif', writer='ffmpeg', fps=60)
    
    plt.show()

animate_comparison()