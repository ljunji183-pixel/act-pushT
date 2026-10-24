import torch
from torch import nn
from torchvision.models import resnet18


def reparameterize(mu, logvar):
    """重参数化采样：z = mu + std * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class CVAEEncoder(nn.Module):
    """训练时编码器：根据当前状态和真实动作序列推断隐变量 z 的高斯分布 (mu, logvar)"""

    def __init__(self, chunk_size=16, latent_dim=32, d_model=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_proj = nn.Linear(2, d_model)
        self.action_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size + 2, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, state, actions, action_is_pad=None):
        batch_size = state.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)       # [B, 1, 256]
        state_tokens = self.state_proj(state).unsqueeze(1)           # [B, 1, 256]
        action_tokens = self.action_proj(actions)                    # [B, K, 256]

        tokens = torch.cat([cls_tokens, state_tokens, action_tokens], dim=1)
        tokens = tokens + self.pos_embed

        padding_mask = None
        if action_is_pad is not None:
            prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=state.device)
            padding_mask = torch.cat([prefix_mask, action_is_pad.bool()], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        summary = encoded[:, 0, :]                                   # 取 CLS token
        mu = self.to_mu(summary)
        logvar = self.to_logvar(summary).clamp(-10.0, 2.0)
        return mu, logvar


class ObservationEncoder(nn.Module):
    """观测编码器：融合 ResNet18 图像特征、机器人状态与隐变量 z"""

    def __init__(self, latent_dim=32, d_model=256, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)

        # 去掉全局池化和分类头，保留 [B, 512, 3, 3] 空间特征
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.input_proj = nn.Linear(512, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)

        self.state_proj = nn.Linear(2, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, images, states, z):
        feat = self.features(images)                                 # [B, 512, 3, 3]
        img_tokens = feat.flatten(2).transpose(1, 2)                 # [B, 9, 512]
        img_tokens = self.input_proj(img_tokens) + self.pos_embed    # [B, 9, 256]

        state_token = self.state_proj(states).unsqueeze(1)           # [B, 1, 256]
        latent_token = self.latent_proj(z).unsqueeze(1)              # [B, 1, 256]

        tokens = torch.cat([latent_token, state_token, img_tokens], dim=1)
        return self.transformer_encoder(tokens)                      # [B, 11, 256]


class ACTPolicy(nn.Module):
    """精简版 ACT (Action Chunking with Transformers)"""

    def __init__(self, chunk_size=16, latent_dim=32, d_model=256, pretrained_backbone=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        self.cvae_encoder = CVAEEncoder(chunk_size, latent_dim, d_model)
        self.observation_encoder = ObservationEncoder(latent_dim, d_model, pretrained_backbone)

        # 动作查询向量 + 全局隐变量偏置（[B, 1, 256]]
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.z_proj = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.action_head = nn.Linear(d_model, 2)

    def forward(self, images, states, actions=None, action_is_pad=None):
        batch_size = images.size(0)

        if actions is not None:
            mu, logvar = self.cvae_encoder(states, actions, action_is_pad)
            z = reparameterize(mu, logvar)
        else:
            mu, logvar = None, None
            z = states.new_zeros(batch_size, self.latent_dim)

        memory = self.observation_encoder(images, states, z)
        queries = self.action_queries.expand(batch_size, -1, -1) + self.z_proj(z).unsqueeze(1)
        decoded = self.decoder(tgt=queries, memory=memory)
        predicted_actions = self.action_head(decoded)

        return predicted_actions, mu, logvar