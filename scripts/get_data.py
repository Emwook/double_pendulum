from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import torch as tr

#loading the data in the torch tensor format
def get_data(trial_num = 3):
    data_dir = pjoin('data', 'Video_Tracking_Data', f'Trial{trial_num}')
    theta1_deg = np.load(pjoin(data_dir, 'DPmean_data_RB0.npy')).flatten()
    theta2_deg = np.load(pjoin(data_dir, 'DPmean_data_RB1.npy')).flatten()

    theta1 = np.radians(theta1_deg)
    theta2= np.radians(theta2_deg)

    min_len = min(len(theta1), len(theta2))
    theta1, theta2 = theta1[:min_len], theta2[:min_len]

    numpy_data = np.column_stack([theta1, theta2])

    tensor_data = tr.tensor(numpy_data, dtype=tr.float32)
    return tensor_data


def plot_data(data):
    plt.figure(figsize=(12, 5))
    th1, th2 = data[:,0], data[:,1]
    L1, L2 = 0.2, 0.2
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(th1, th2, lw=0.5, color='purple')
    plt.title("Phase Space (Radians)")
    plt.xlabel("Theta 1")
    plt.ylabel("Theta 2")

    plt.subplot(1, 2, 2)
    plt.plot(x1, y1, lw=0.7, color='blue', label="Joint (L1)")
    plt.plot(x2, y2, lw=0.5, color='red', alpha=0.6, label="Tip (L2)")
    plt.axis('equal')
    plt.title("Physical Trajectory (Meters)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_source():
    tensor_data = get_data()
    plot_data(tensor_data.numpy())
   
    
# plot_source()