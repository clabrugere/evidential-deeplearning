import torch
from torch import nn
import torch.nn.functional as F


class EDLClassifier(nn.Module):
    def __init__(self, encoder, dim_encoder_out, dim_hidden, num_classes, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.encoder = encoder
        self.fc = nn.Linear(dim_encoder_out, dim_hidden)
        self.dropout = nn.Dropout(dropout)
        self.projection_head = nn.Linear(dim_hidden, num_classes, bias=False)

    def forward(self, x):
        # instead of using regular softmax or sigmoid to output a probability distribution over the classes,
        # we output a positive vector, using a softplus on the logits, as the evidence over the classes
        x = self.encoder(x)

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.projection_head(x)

        return F.softplus(x)

    @torch.inference_mode()
    def predict(self, x, return_uncertainty=True):
        evidences = self(x)
        # alphas are the parameters of the Dirichlet distribution that models the probability distribution over the
        # class probabilities and strength is the Dirichlet strength
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        if return_uncertainty:
            total_uncertainty = self.num_classes / strength
            beliefs = evidences / strength
            return probabilities, total_uncertainty, beliefs
        else:
            return probabilities
