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
        alphas = evidences + 1
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        if return_uncertainty:
            total_uncertainty = self.num_classes / strength
            beliefs = evidences / strength
            return probabilities, total_uncertainty, beliefs
        else:
            return probabilities


class TypeIIMaximumLikelihoodLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Dirichlet distribution D(p|alphas) is used as prior on the likelihood of Multi(y|p)."""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)

        loss = torch.sum(labels * (torch.log(strength) - torch.log(alphas)), dim=-1)

        return torch.mean(loss)


class CEBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Bayes risk is the maximum cost of making incorrect estimates, taking a cost function assigning a penalty of
        making an incorrect estimate and summing it over all possible outcomes. Here the cost function is the Cross Entropy.
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strengths = torch.sum(alphas, dim=-1, keepdim=True)

        loss = torch.sum(labels * (torch.digamma(strengths) - torch.digamma(alphas)), dim=-1)

        return torch.mean(loss)


class SSBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Same as CEBayesRiskLoss but here the cost function is the sum of squares instead."""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        error = (labels - probabilities) ** 2
        variance = probabilities * (1 - probabilities) / (strength + 1)

        loss = torch.sum(error + variance, dim=-1)

        return torch.mean(loss)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Acts as a regularization term to shrink towards zero the evidence of samples that cannot be correctly classified"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        num_classes = evidences.size(-1)
        alphas = evidences + 1.0
        alphas_tilde = labels + (1 - labels) * alphas
        strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

        # lgamma is the log of the gamma function
        first_term = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32))
            - torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True
        )
        loss = torch.mean(first_term + second_term)

        return loss
