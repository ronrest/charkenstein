import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, in_size, embed_size, h_size, out_size, n_layers=1, dropout=0.7):
        super(Model, self).__init__()
        
        self.in_size = in_size
        self.embed_size = embed_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.alpha = 0.001       # learning rate
        
        self.embeddings = nn.Embedding(in_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=h_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout
                            )
        self.classifier = nn.Linear(h_size, out_size)

        # SPECIFY LOSS AND OPTIMIZER FUNCTIONS
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, input, hidden):
        timesteps = 1 # keep timesteps fixed to 1 for generative tasks
        
        input = self.embeddings(input)
        # Reshape input dimensions to (batch_size, timesteps, embed_size)
        input = input.view(self.batch_size, timesteps, self.embed_size)
        
        output, hidden = self.lstm(input, hidden)
        output = self.classifier(output.view(self.batch_size, self.h_size))
        return output, hidden
    
    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        # Note batch is always second index for hidden EVEN when batch_first
        # argument was set for the LSTM.
        return (Variable(torch.zeros(self.n_layers, batch_size, self.h_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.h_size)))

    def update_learning_rate(self, alpha):
        """ Updates the learning rate without resetting momentum."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = alpha
        self.alpha = alpha
        

