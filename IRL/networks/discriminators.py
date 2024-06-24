# discriminators for adversarial IRL
import torch
import torch.nn as nn
import torch.nn.functional as F

class AIRLDiscriminator(nn.Module):
    def __init__(self,
                 state_dim,
                 gamma=0.99,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)
                 ):
        super().__init__()
        
        self.gamma = gamma
        self.g_predictor_layers = nn.ModuleList()
        self.v_predictor_layers = nn.ModuleList()
        output_dim = 1
        for i in range(len(hidden_units_r)):
            if i == 0:
                self.g_predictor_layers.append(nn.Linear(state_dim, hidden_units_r[i]))
            else:
                self.g_predictor_layers.append(nn.Linear(hidden_units_r[i-1], hidden_units_r[i]))
            self.g_predictor_layers.append(hidden_activation_r)
        self.g_predictor_layers.append(nn.Linear(hidden_units_r[-1], output_dim))

        for i in range(len(hidden_units_v)):
            if i == 0:
                self.v_predictor_layers.append(nn.Linear(state_dim, hidden_units_v[i]))
            else:
                self.v_predictor_layers.append(nn.Linear(hidden_units_v[i-1], hidden_units_v[i]))
            self.v_predictor_layers.append(hidden_activation_v)
        self.v_predictor_layers.append(nn.Linear(hidden_units_v[-1], output_dim))

        self.g_predictor = nn.Sequential(*self.g_predictor_layers)
        self.v_predictor = nn.Sequential(*self.v_predictor_layers)

    def forward(self, states, dones, log_pis, next_states):
        g = self.g_predictor(states)
        v = self.v_predictor(states)
        next_v = self.v_predictor(next_states)
        pred_f_theta = g + self.gamma * next_v * (1 - dones) - v
        output = pred_f_theta - log_pis
        return output

    def predict_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            disc_result = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-disc_result)

