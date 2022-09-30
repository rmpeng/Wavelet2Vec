import torch
import torch.nn as nn
from base_models.base_transformer import Transformer
from utils.utils import get_1d_sincos_pos_embed
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )

    def forward(self, x):
        return self.proj(x)


class WaveletTimeTransformer(nn.Module):
    def __init__(self, org_signal_length=512, org_channel=20, patch_size=40, pool='mean', in_chans=1, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4, dropout=0.2, num_classes=4):
        super().__init__()
        signal_length = org_signal_length * org_channel
        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.data_rebuild = Rearrange('b c h w -> b h (c w)')
        self.patch_embed_delta = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_embed_theta = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_embed_alpha = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_embed_beta = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_embed_gamma = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_embed_upper = PatchEmbed(patch_size, in_chans, embed_dim)

        self.cls_token_delta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_theta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_alpha = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_beta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_gamma = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_upper = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_delta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_theta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_alpha = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_beta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_gamma = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_upper = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.dropout_delta = nn.Dropout(dropout)
        self.dropout_theta = nn.Dropout(dropout)
        self.dropout_alpha = nn.Dropout(dropout)
        self.dropout_beta = nn.Dropout(dropout)
        self.dropout_gamma = nn.Dropout(dropout)
        self.dropout_upper = nn.Dropout(dropout)

        self.blocks_delta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_theta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_alpha = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_beta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_gamma = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_upper = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)


        # 讨论空间
        self.classifier_delta = nn.Linear(embed_dim, num_classes)
        self.classifier_theta = nn.Linear(embed_dim, num_classes)
        self.classifier_alpha = nn.Linear(embed_dim, num_classes)
        self.classifier_beta = nn.Linear(embed_dim, num_classes)
        self.classifier_gamma = nn.Linear(embed_dim, num_classes)
        self.classifier_upper = nn.Linear(embed_dim, num_classes)

        self.cls_weight_delta = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)
        self.cls_weight_theta = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)
        self.cls_weight_alpha = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)
        self.cls_weight_beta = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)
        self.cls_weight_gamma = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)
        self.cls_weight_upper = nn.Parameter(0.1667 * torch.ones(1,), requires_grad=True)

        self.cls_weight_delta.requires_grad_ = True
        self.cls_weight_theta.requires_grad_ = True
        self.cls_weight_alpha.requires_grad_ = True
        self.cls_weight_beta.requires_grad_ = True
        self.cls_weight_gamma.requires_grad_ = True
        self.cls_weight_upper.requires_grad_ = True


        self.initialize_weights()

    def initialize_weights(self):
        pos_embed_delta = get_1d_sincos_pos_embed(self.pos_embed_delta.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_delta.data.copy_(torch.from_numpy(pos_embed_delta).float().unsqueeze(0))
        pos_embed_theta = get_1d_sincos_pos_embed(self.pos_embed_theta.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_theta.data.copy_(torch.from_numpy(pos_embed_theta).float().unsqueeze(0))
        pos_embed_alpha = get_1d_sincos_pos_embed(self.pos_embed_alpha.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_alpha.data.copy_(torch.from_numpy(pos_embed_alpha).float().unsqueeze(0))
        pos_embed_beta = get_1d_sincos_pos_embed(self.pos_embed_beta.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_beta.data.copy_(torch.from_numpy(pos_embed_beta).float().unsqueeze(0))
        pos_embed_gamma = get_1d_sincos_pos_embed(self.pos_embed_gamma.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_gamma.data.copy_(torch.from_numpy(pos_embed_gamma).float().unsqueeze(0))
        pos_embed_upper = get_1d_sincos_pos_embed(self.pos_embed_upper.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed_upper.data.copy_(torch.from_numpy(pos_embed_upper).float().unsqueeze(0))

    def forward(self,  x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper):

        # embed patches
        x_delta = self.data_rebuild(x_delta)
        x_theta = self.data_rebuild(x_theta)
        x_alpha = self.data_rebuild(x_alpha)
        x_beta = self.data_rebuild(x_beta)
        x_gamma = self.data_rebuild(x_gamma)
        x_upper = self.data_rebuild(x_upper)

        x_delta = self.patch_embed_delta(x_delta)
        x_theta = self.patch_embed_theta(x_theta)
        x_alpha = self.patch_embed_alpha(x_alpha)
        x_beta = self.patch_embed_beta(x_beta)
        x_gamma = self.patch_embed_gamma(x_gamma)
        x_upper = self.patch_embed_upper(x_upper)

        # add pos embed w/o cls token
        x_delta = x_delta + self.pos_embed_delta[:, 1:, :]
        x_theta = x_theta + self.pos_embed_theta[:, 1:, :]
        x_alpha = x_alpha + self.pos_embed_alpha[:, 1:, :]
        x_beta = x_beta + self.pos_embed_beta[:, 1:, :]
        x_gamma = x_gamma + self.pos_embed_gamma[:, 1:, :]
        x_upper = x_upper + self.pos_embed_upper[:, 1:, :]

        # append cls token
        cls_token_delta = self.cls_token_delta + self.pos_embed_delta[:, :1, :]
        cls_tokens_delta = cls_token_delta.expand(x_delta.shape[0], -1, -1)
        x_delta = torch.cat((cls_tokens_delta, x_delta), dim=1)

        # append cls token
        cls_token_theta = self.cls_token_theta + self.pos_embed_theta[:, :1, :]
        cls_tokens_theta = cls_token_theta.expand(x_theta.shape[0], -1, -1)
        x_theta = torch.cat((cls_tokens_theta, x_theta), dim=1)

        # append cls token
        cls_token_alpha = self.cls_token_alpha + self.pos_embed_alpha[:, :1, :]
        cls_tokens_alpha = cls_token_alpha.expand(x_alpha.shape[0], -1, -1)
        x_alpha = torch.cat((cls_tokens_alpha, x_alpha), dim=1)

        # append cls token
        cls_token_beta = self.cls_token_beta + self.pos_embed_beta[:, :1, :]
        cls_tokens_beta = cls_token_beta.expand(x_beta.shape[0], -1, -1)
        x_beta = torch.cat((cls_tokens_beta, x_beta), dim=1)

        # append cls token
        cls_token_gamma = self.cls_token_gamma + self.pos_embed_gamma[:, :1, :]
        cls_tokens_gamma = cls_token_gamma.expand(x_gamma.shape[0], -1, -1)
        x_gamma = torch.cat((cls_tokens_gamma, x_gamma), dim=1)

        # append cls token
        cls_token_upper = self.cls_token_upper + self.pos_embed_upper[:, :1, :]
        cls_tokens_upper = cls_token_upper.expand(x_upper.shape[0], -1, -1)
        x_upper = torch.cat((cls_tokens_upper, x_upper), dim=1)

        # apply Transformer blocks
        x_delta = self.dropout_delta(x_delta)
        x_delta = self.blocks_delta(x_delta)

        x_theta = self.dropout_theta(x_theta)
        x_theta = self.blocks_theta(x_theta)

        x_alpha = self.dropout_alpha(x_alpha)
        x_alpha = self.blocks_alpha(x_alpha)

        x_beta = self.dropout_beta(x_beta)
        x_beta = self.blocks_beta(x_beta)

        x_gamma = self.dropout_gamma(x_gamma)
        x_gamma = self.blocks_gamma(x_gamma)

        x_upper = self.dropout_upper(x_upper)
        x_upper = self.blocks_upper(x_upper)

        # using token feature as classification head or avgerage pooling for all feature
        x_delta = x_delta.mean(dim=1) if self.pool == 'mean' else x_delta[:, 0]
        x_theta = x_theta.mean(dim=1) if self.pool == 'mean' else x_theta[:, 0]
        x_alpha = x_alpha.mean(dim=1) if self.pool == 'mean' else x_alpha[:, 0]
        x_beta = x_beta.mean(dim=1) if self.pool == 'mean' else x_beta[:, 0]
        x_gamma = x_gamma.mean(dim=1) if self.pool == 'mean' else x_gamma[:, 0]
        x_upper = x_upper.mean(dim=1) if self.pool == 'mean' else x_upper[:, 0]


        # classify
        pred_delta = self.classifier_delta(x_delta)
        pred_theta = self.classifier_theta(x_theta)
        pred_alpha = self.classifier_alpha(x_alpha)
        pred_beta = self.classifier_beta(x_beta)
        pred_gamma = self.classifier_gamma(x_gamma)
        pred_upper = self.classifier_upper(x_upper)


        # pred = (pred_beta + pred_upper + pred_gamma + pred_alpha + pred_theta + pred_delta) / 6
        pred = pred_delta * self.cls_weight_delta +\
               pred_theta * self.cls_weight_theta +\
               pred_alpha * self.cls_weight_alpha +\
               pred_beta * self.cls_weight_beta+\
               pred_gamma * self.cls_weight_gamma +\
               pred_upper * self.cls_weight_upper


        return pred
