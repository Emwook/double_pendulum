import torch as tr
from gru import Gru_Cell

# LOSS FUNCTION
# Loss = MSE(theta_pred, theta_actual) + lambda*L^2
#
# MSE = 1/n sum((theta_pred - theta_actual)^2)
#
# L^2 = L^2_1 + L^2_2
#
# L_1 = 4/3*l*theta_dd_1 + 1/2*l*theta_dd_2*cos(theta_1-theta_2) 
#       + 1/2*l*theta^2_d_2 * sin(theta_1 - theta_2) + 3/2g*sin(theta_1)
#
# L_2 = 1/3*l*theta_dd_2 + 1/2*l*theta_dd_1*cos(theta_1-theta_2) 
#    - 1/2*l*theta^2_d_1*sin(theta_1 - theta_2) + 1/2g*sin(theta_2)

class PINN_Network(tr.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru_cell = Gru_Cell(input_size, hidden_size)

        self.W_out = tr.nn.Parameter(tr.randn(hidden_size, input_size))
        self.b_out = tr.nn.Parameter(tr.zeros(input_size))

    def pinn_loss_function(self,theta_pred, theta_actual, x_window, dt, l, g, lambda_w, PINN = True):
        mse_loss = tr.mean((theta_pred - theta_actual)**2)

        #getting previous physical params by finite differences
        theta_t      = x_window[:, -1, :]
        theta_t_prev = x_window[:, -2, :]
        omega_old  = (theta_t - theta_t_prev) / dt
        omega_pred = (theta_pred - theta_t) / dt
        epsilon_pred = (omega_pred - omega_old) / dt

        th1 = theta_pred[:,0]
        th2 = theta_pred[:,1]
        om1 = omega_pred[:,0]
        om2 = omega_pred[:,1]
        ep1 = epsilon_pred[:,0]
        ep2 = epsilon_pred[:,1]

        #calculating the lagrangian based loss function
        L_1 = (4/3)*l*ep1 + (1/2)*l*ep2*tr.cos(th1-th2) + (1/2)*l*om2**2 * tr.sin(th1 - th2) + (3/2)* g * tr.sin(th1)
        L_2 = (1/3)*l*ep2 + (1/2)*l*ep1*tr.cos(th1-th2) - (1/2)*l*om1**2 * tr.sin(th1 - th2) + (1/2)* g * tr.sin(th2)

        L = tr.mean(L_1**2) + tr.mean(L_2**2)

        loss = mse_loss
        if(PINN == True):
            loss += lambda_w*L
        return loss, mse_loss, L
    
    def forward(self, x_window):
        batch_size = x_window.shape[0]      #window amount
        seq_len = x_window.shape[1]    #window size

        #starting state - blank
        h = tr.zeros(batch_size, self.gru_cell.W.shape[1], device=x_window.device)

        for t in range(seq_len):
            x_t = x_window[:, t, :]
            h = self.gru_cell.get_new_h(x_t, h)

        # translating the window sized vector to the [theta1, theta2] output
        next_step_prediction = (h @ self.W_out) + self.b_out

        return next_step_prediction

