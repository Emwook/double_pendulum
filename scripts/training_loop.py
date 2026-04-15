from pinn_network import PINN_Network
import torch as tr 
from get_data import get_data
from torch.utils.data import DataLoader
from dataset import *
import tqdm

def training_loop():
    all_trails = []
    trial_amount = 3
    stride = 10

    for i in range(trial_amount):
        tensor_data = get_data(i+1,stride)
        all_trails.append(tensor_data)


    seq_len = 200
    batch_size = 32

    data = tr.cat(all_trails, dim=0)
    dataset = Pendulum_Dataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    dt = 0.001*stride
    l = 0.2
    g = 9.81
    lambda_w = 0.05
    learning_rate = 0.001
    epochs = 100

    input_size  = 2
    hidden_size = 32

    model = PINN_Network(input_size,hidden_size)

    #using adam to fit the params, idea for the future: build own adam
    optimizer = tr.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_bar = tqdm.tqdm(range(epochs), desc="training", unit="epoch")
    for epoch in epoch_bar:
        if(epoch<5):
            current_lambda = 0
        else:
            current_lambda = lambda_w
        
        for x_window, y_actual in dataloader: 
            optimizer.zero_grad()
            
            theta_pred = model(x_window)
            total_loss, mse, phys = model.pinn_loss_function(
                theta_pred, y_actual, x_window, dt, l, g, current_lambda
            )

            total_loss.backward()
            tr.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        print(f"epoch {epoch}, total loss: {total_loss.item():.4f}, mse: {mse.item():.4f}, pinn part: {phys.item():.4f}")
    
    tr.save(model.state_dict(), "results/pinn_pendulum_v2.pth")
    print("model saved")

training_loop()