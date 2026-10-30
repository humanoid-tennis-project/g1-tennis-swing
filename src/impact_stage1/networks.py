"""Actor bottleneck with no direct reference-to-decoder bypass."""
import torch
from torch import nn


def mlp(dimensions):
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dimensions)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class SkillBottleneck(nn.Module):
    """Input [reference63, proprioception87, task10]; output actions27.

    The first 63 contain reference joints54 plus reference-anchor position/orientation9.
    They must not bypass the encoder into the reusable decoder.
    """
    def __init__(self, reference_dim=63, context_dim=87, task_dim=10, latent_dim=8,
                 action_dim=27, hidden=(512, 256, 128)):
        super().__init__()
        self.reference_dim, self.context_dim = reference_dim, context_dim
        self.input_dim = reference_dim+context_dim+task_dim
        self.encoder = mlp((self.input_dim, *hidden, latent_dim))
        self.decoder = mlp((context_dim+latent_dim, *hidden, action_dim))

    def encode(self, normalized_observations):
        return torch.tanh(self.encoder(normalized_observations))

    def lower(self, normalized_context, z):
        return self.decoder(torch.cat((normalized_context, z), dim=-1))

    def forward(self, normalized_observations):
        z = self.encode(normalized_observations)
        context = normalized_observations[..., self.reference_dim:self.reference_dim+self.context_dim]
        return self.lower(context, z)


def unit_vector(v):
    return v/torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def target_rewards(position, velocity, normal, target_p, target_v, target_n,
                   position_std=.15, velocity_std=2., normal_std=.35):
    """Axial racket normal: n and -n describe the same physical plane."""
    pe = ((position-target_p)**2).sum(-1)
    ve = ((velocity-target_v)**2).sum(-1)
    cosine = (unit_vector(normal)*unit_vector(target_n)).sum(-1).abs().clamp(0, 1)
    ae = torch.acos(cosine)**2
    return torch.stack((torch.exp(-pe/position_std**2), torch.exp(-ve/velocity_std**2),
                        torch.exp(-ae/normal_std**2)), dim=-1)
