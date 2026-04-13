from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt

trial_num = 3

def get_data():
    data_dir = pjoin('data', 'Video_Tracking_Data', f'Trial{trial_num}')
    theta1_deg = np.load(pjoin(data_dir, 'DPmean_data_RB0.npy')).flatten()
    theta2_deg = np.load(pjoin(data_dir, 'DPmean_data_RB1.npy')).flatten()

    theta1 = np.radians(theta1_deg)
    theta2= np.radians(theta2_deg)

    min_len = min(len(theta1), len(theta2))
    theta1, theta2 = theta1[:min_len], theta2[:min_len]

    L1, L2 = 0.5, 0.5
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    data = np.column_stack([theta1, theta2, x1, y1, x2, y2])
    return data

def get_velocity(data, dt=0.001):
    velocity = (data[1:] - data[:-1])/(dt)
    velocity = np.vstack([velocity, velocity[-1:]])
    return velocity

def plot_data(data):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[:,0], data[:,1], lw=0.5)
    plt.title("Phase Space")
    plt.xlabel("Theta 1")
    plt.ylabel("Theta 2")

    plt.subplot(1, 2, 2)
    plt.plot(data[:,2], data[:,3], lw=0.5, color='blue', label="Joint")
    plt.plot(data[:,4], data[:,5], lw=0.5, color='red', alpha=0.5, label="Tip")
    plt.axis('equal')
    plt.title("Physical Trajectory")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    data = get_data()
    velocity  = get_velocity(data)
    plot_data(data)


# if '__name__' == '_main_':

main()