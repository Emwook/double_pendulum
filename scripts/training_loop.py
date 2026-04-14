from pinn_network import PINN_Network
import torch as tr 
from get_data import get_data
from torch.utils.data import DataLoader
from dataset import *
import tqdm

def training_loop():
    all_trails = []
    trial_amount = 3

    for i in range(trial_amount):
        tensor_data = get_data(i+1)
        all_trails.append(tensor_data)


    seq_len = 100
    batch_size = 64

    data = tr.cat(all_trails, dim=0)
    dataset = Pendulum_Dataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    dt = 0.001
    l = 0.2
    g = 9.81
    lambda_w = 0
    learning_rate = 0.01
    epochs = 5

    input_size  = 2
    hidden_size = 32

    model = PINN_Network(input_size,hidden_size)

    #using adam to fit the params, idea for the future: build own adam
    optimizer = tr.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_bar = tqdm.tqdm(range(epochs), desc="training", unit="epoch")
    for epoch in epoch_bar:
        for x_window, y_actual in dataloader: 
            optimizer.zero_grad()
            
            theta_pred = model(x_window)
            total_loss, mse, phys = model.pinn_loss_function(
                theta_pred, y_actual, x_window, dt, l, g, lambda_w, True
            )

            total_loss.backward()
            tr.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        print(f"epoch {epoch}, total loss: {total_loss.item():.4f}, mse: {mse.item():.4f}, pinn part: {phys.item():.4f}")
    
    tr.save(model.state_dict(), "results/pinn_pendulum_v1.pth")
    print("model saved")

training_loop()