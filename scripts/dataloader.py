import torch as tr
import torch.utils.data as trd
from dataset import Pendulum_Dataset

total_frames = 10000 
raw_sensor_data = tr.randn(total_frames, 2) 

window_size = 50
dataset = Pendulum_Dataset(raw_sensor_data, seq_len=window_size)

batch_size = 64
dataloader = trd.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True 
)
