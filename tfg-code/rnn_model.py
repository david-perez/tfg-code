import torch


class RNNModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(RNNModel, self).__init__()

        self.rnn = torch.nn.RNN(
            input_size=D_in,
            hidden_size=H,
            batch_first=True,  # Input is provided with shape (batch, seq_len, input_size)
            num_layers=1
        )  # TODO: Dropout

        self.out = torch.nn.Linear(H, D_out)

    def forward(self, x):
        r_out, _ = self.rnn(x)  # r_out has shape (batch, seq_len, H)
        last_hidden = r_out[:, -1, :]  # Get hidden state from last unit of the sequence, all batches.
        return self.out(last_hidden)