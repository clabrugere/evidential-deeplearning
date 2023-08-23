import torch
from torch import nn


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
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)

        loss = torch.sum(error + variance, dim=-1)

        return torch.mean(loss)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Acts as a regularization term to shrink towards zero the evidence of samples that cannot be correctly classified"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        num_classes = evidences.size(-1)
        alphas = evidences + 1.0
        alphas_tilde = labels + (1.0 - labels) * alphas
        strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

        # lgamma is the log of the gamma function
        first_term = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32))
            - torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True
        )
        loss = torch.mean(first_term + second_term)

        return loss
