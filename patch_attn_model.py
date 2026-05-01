import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from attention_pooling import AttentionPooling


# 🔥 NEW: Gradient Reversal Layer
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


# 🔥 NEW: Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # source vs target
        )

    def forward(self, x):
        return self.net(x)


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.squeeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn = self.fc(x)
        attn = torch.softmax(attn, dim=1)
        x = x * attn
        return x


class PatchAttentionCLModel(nn.Module):
    def __init__(self, embed_dim: int = 256, attr_dim: int = 0, attn_hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = ResNet34Encoder()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.embed_dim)
        )

        self.attn_pool = AttentionPooling(
            embed_dim=self.embed_dim,
            hidden_dim=attn_hidden_dim,
            attr_dim=attr_dim
        )

        self.temporal_attention = TemporalAttention(self.embed_dim)
        self.temporal_encoder = TemporalEncoder(self.embed_dim)

        if attr_dim > 0:
            self.attr_mlp = nn.Sequential(
                nn.Linear(attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.embed_dim)
            )
            self.final_dim = 3 * self.embed_dim
        else:
            self.attr_mlp = None
            self.final_dim = 2 * self.embed_dim

        # 🔥 NEW: domain discriminator (inactive unless used)
        self.domain_disc = DomainDiscriminator(self.final_dim)

        self.fusion = nn.Identity()

    def encode_patches(self, patches):
        if patches.dim() == 5:
            B, N, C, H, W = patches.shape
            flat = patches.view(B * N, C, H, W)
            f = self.encoder(flat)
            proj = self.projector(f)
            proj = F.normalize(proj, dim=1)
            return proj.view(B, N, -1)

        elif patches.dim() == 4:
            f = self.encoder(patches)
            proj = self.projector(f)
            return proj

        else:
            raise ValueError(f"Invalid shape {patches.shape}")

    def forward(self, patches, batch_size, num_patches, attrs=None):
        B, N, C, H, W = patches.shape
        flat = patches.view(B * N, C, H, W)

        proj = self.encode_patches(flat)
        proj = proj.view(B, N, -1)

        pooled = self.attn_pool(proj, attrs)

        proj_attn = self.temporal_attention(proj)
        temporal_feat = self.temporal_encoder(proj_attn)

        z = torch.cat([pooled, temporal_feat], dim=1)

        if self.attr_mlp is not None and attrs is not None:
            a = self.attr_mlp(attrs)
            z = torch.cat([z, a], dim=1)

        z = F.normalize(z, dim=1)

        return self.fusion(z)

    # 🔥 OPTIONAL (for DANN later)
    def forward_domain(self, z, lambda_=0.5):
        z_rev = GRL.apply(z, lambda_)
        return self.domain_disc(z_rev)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        positives = torch.cat([
            torch.diag(sim, N),
            torch.diag(sim, -N)
        ], dim=0)

        log_prob = torch.logsumexp(sim, dim=1)

        loss = -positives + log_prob
        return loss.mean()