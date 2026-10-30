"""Use upstream RSL-RL PPO unchanged, replacing only the Actor mean network."""
from rsl_rl.models import MLPModel
from .networks import SkillBottleneck


class ImpactActor(MLPModel):
    def __init__(self, *args, latent_dim=8, reference_dim=63, context_dim=87, task_dim=10, **kwargs):
        hidden = kwargs.get('hidden_dims', (512, 256, 128))
        super().__init__(*args, **kwargs)
        expected = reference_dim+context_dim+task_dim
        if self.obs_dim != expected:
            raise ValueError(f'Expected Actor input {expected}, got {self.obs_dim}')
        output_dim = kwargs.get('output_dim', args[3] if len(args) > 3 else None)
        if output_dim is None:
            raise ValueError('Explicit output_dim is required')
        if self.distribution is not None and self.distribution.input_dim != output_dim:
            raise ValueError('State-dependent distribution output is not supported')
        self.mlp = SkillBottleneck(reference_dim, context_dim, task_dim, latent_dim, output_dim, hidden)

    def get_latent(self, obs, masks=None, hidden_state=None):
        import torch
        return self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
