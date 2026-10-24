"""ACT with gated sinusoidal encoding on action-chunk time positions."""
import math

import torch
from torch import nn
from torchvision.models import resnet18


def sinusoidal_positions(length, dimension, device=None, dtype=None):
    """Classic Transformer sin/cos positions, returned as [1, length, dimension]."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * div_term)
    result[:, 1::2] = torch.cos(position * div_term[: result[:, 1::2].shape[1]])
    return result.unsqueeze(0).to(dtype=dtype or torch.float32)


class CVAEEncoder(nn.Module):
    def __init__(self, chunk_size=16, latent_dim=32, d_model=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_proj = nn.Linear(2, d_model)
        self.action_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size + 2, d_model) * 0.02)
        self.temporal_scale_raw = nn.Parameter(torch.tensor(0.0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, state, actions, action_is_pad=None):
        batch = state.size(0)
        tokens = torch.cat((
            self.cls_token.expand(batch, -1, -1),
            self.state_proj(state).unsqueeze(1),
            self.action_proj(actions),
        ), dim=1)
        tokens = tokens + self.pos_embed
        time_pe = sinusoidal_positions(
            self.chunk_size, tokens.size(-1), device=tokens.device, dtype=tokens.dtype
        )
        tokens = torch.cat((
            tokens[:, :2],
            tokens[:, 2:] + torch.tanh(self.temporal_scale_raw) * time_pe,
        ), dim=1)
        padding_mask = None
        if action_is_pad is not None:
            padding_mask = torch.cat((
                torch.zeros(batch, 2, dtype=torch.bool, device=state.device),
                action_is_pad.bool(),
            ), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        summary = encoded[:, 0]
        return self.to_mu(summary), self.to_logvar(summary).clamp(-10.0, 2.0)


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)


class ObservationEncoder(nn.Module):
    def __init__(self, latent_dim=32, d_model=256, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.input_proj = nn.Linear(512, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)
        self.state_proj = nn.Linear(2, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, images, states, z):
        feat = self.features(images)
        img_tokens = feat.flatten(2).transpose(1, 2)
        img_tokens = self.input_proj(img_tokens) + self.pos_embed
        return self.transformer_encoder(torch.cat((
            self.latent_proj(z).unsqueeze(1),
            self.state_proj(states).unsqueeze(1),
            img_tokens,
        ), dim=1))


class ACTPolicy(nn.Module):
    """Checkpoint-compatible ACT plus bounded, learnable sinusoidal time signals."""
    def __init__(self, chunk_size=16, latent_dim=32, d_model=256, pretrained_backbone=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.cvae_encoder = CVAEEncoder(chunk_size, latent_dim, d_model)
        self.observation_encoder = ObservationEncoder(latent_dim, d_model, pretrained_backbone)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.decoder_temporal_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.z_proj = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.action_head = nn.Linear(d_model, 2)

    def forward(self, images, states, actions=None, action_is_pad=None):
        batch = images.size(0)
        if actions is not None:
            mu, logvar = self.cvae_encoder(states, actions, action_is_pad)
            z = reparameterize(mu, logvar)
        else:
            mu = logvar = None
            z = states.new_zeros(batch, self.latent_dim)
        memory = self.observation_encoder(images, states, z)
        queries = self.action_queries.expand(batch, -1, -1)
        queries = queries + self.z_proj(z).unsqueeze(1)
        time_pe = sinusoidal_positions(
            self.chunk_size, queries.size(-1), device=queries.device, dtype=queries.dtype
        )
        queries = queries + torch.tanh(self.decoder_temporal_scale_raw) * time_pe
        return self.action_head(self.decoder(tgt=queries, memory=memory)), mu, logvar
