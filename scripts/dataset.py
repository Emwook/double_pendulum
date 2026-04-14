import torch.utils.data as trd

#check the lengths to fit the window lengths

class Pendulum_Dataset(trd.Dataset):
    def __init__(self, raw_data, seq_len):
        super().__init__()
        self.data = raw_data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, index):
        x_window = self.data[index : index + self.seq_len]
        y_actual = self.data[index + self.seq_len]

        return x_window, y_actual