import torch
import torch.nn as nn


class BatchedLSTM(nn.Module):
    def __init__(self, C, H, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.C = C
        self.H = H
        self.forget_gate = nn.Conv1d(in_channels=C + H, out_channels=H, kernel_size=1)

        self.input_gate = nn.Conv1d(
            in_channels=C + H,
            out_channels=H,
            kernel_size=1,
        )

        self.output_gate = nn.Conv1d(
            in_channels=C + H,
            out_channels=H,
            kernel_size=1,
        )

        self.tanh_gate = nn.Conv1d(
            in_channels=C + H,
            out_channels=H,
            kernel_size=1,
        )

    def flatten_parameters(self):
        pass

    def forward(self, x, state):
        """
        Input: (B, F, C) <--- F dimensions is processed independently
        """
        hidden_state, cell_state = state
        # x_hidden = torch.cat([x, hidden_state], dim=-1) # [B, F, C+H]

        x = x.transpose(1, 2)  # [B, C, F]
        x_hidden = torch.cat([x, hidden_state], dim=1)  # [B, C+H, F]

        forget_tensor = self.forget_gate(x_hidden)
        forget_tensor = nn.Sigmoid()(forget_tensor)

        input_tensor = self.input_gate(x_hidden)
        input_tensor = nn.Sigmoid()(input_tensor)

        tanh_tensor = self.tanh_gate(x_hidden)
        tanh_tensor = nn.Tanh()(tanh_tensor)

        output_tensor = self.output_gate(x_hidden)
        output_tensor = nn.Sigmoid()(output_tensor)

        remaining_cell_state = cell_state * forget_tensor
        cell_state_update = input_tensor * tanh_tensor

        new_cell_state = remaining_cell_state + cell_state_update
        new_hidden_state_candidate = new_cell_state
        new_hidden_state_candidate = nn.Tanh()(new_hidden_state_candidate)

        new_hidden_state = new_hidden_state_candidate * output_tensor

        return new_hidden_state.transpose(1, 2), (new_hidden_state, new_cell_state)

    def set_weights(self, state_dict: dict):
        Wi = state_dict["weight_ih_l0"]
        Wh = state_dict["weight_hh_l0"]
        B = state_dict["bias_ih_l0"] + state_dict["bias_hh_l0"]

        K = self.H
        i = 0
        W_i = (
            torch.cat([Wi[i * K : (i + 1) * K], Wh[i * K : (i + 1) * K]], dim=1)
            .unsqueeze(0)
            .permute(1, 2, 0)
        )
        B_i = B[i * K : (i + 1) * K]
        self.input_gate.load_state_dict({"weight": W_i, "bias": B_i})

        i = 1
        W_f = (
            torch.cat([Wi[i * K : (i + 1) * K], Wh[i * K : (i + 1) * K]], dim=1)
            .unsqueeze(0)
            .permute(1, 2, 0)
        )
        B_f = B[i * K : (i + 1) * K]
        self.forget_gate.load_state_dict({"weight": W_f, "bias": B_f})

        i = 2
        W_g = (
            torch.cat([Wi[i * K : (i + 1) * K], Wh[i * K : (i + 1) * K]], dim=1)
            .unsqueeze(0)
            .permute(1, 2, 0)
        )
        B_g = B[i * K : (i + 1) * K]
        self.tanh_gate.load_state_dict({"weight": W_g, "bias": B_g})

        i = 3
        W_o = (
            torch.cat([Wi[i * K : (i + 1) * K], Wh[i * K : (i + 1) * K]], dim=1)
            .unsqueeze(0)
            .permute(1, 2, 0)
        )
        B_o = B[i * K : (i + 1) * K]
        self.output_gate.load_state_dict({"weight": W_o, "bias": B_o})
