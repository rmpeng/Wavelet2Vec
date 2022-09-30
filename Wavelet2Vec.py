from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from base_models.base_transformer import Transformer
from utils.utils import get_1d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )

    def forward(self, x):
        return self.proj(x)

class Wavelet_Transformer(nn.Module):
    def __init__(self, org_signal_length=512, org_channel=20, patch_size=80, in_chans=1, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4, dropout=0.1,
                 decoder_embed_dim=64, decoder_depth=2):
        super().__init__()
        signal_length = org_signal_length * org_channel
        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size
        self.data_rebuild = Rearrange('b c h w -> b h (c w)')

        # --------------------------------------------------------------------------
        # encoder specifics
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


        self.blocks_delta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_theta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_alpha = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_beta = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_gamma = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks_upper = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed_delta = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_theta = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_alpha = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_beta = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_gamma = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_upper = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token_delta = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_theta = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_alpha = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_beta = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_gamma = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_upper = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_delta = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_pos_embed_theta = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_pos_embed_alpha = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_pos_embed_beta = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_pos_embed_gamma = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_pos_embed_upper = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)

        self.decoder_blocks_delta = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_blocks_theta = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_blocks_alpha = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_blocks_beta = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_blocks_gamma = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.decoder_blocks_upper = Transformer(decoder_embed_dim, decoder_depth, num_heads, mlp_ratio, dropout)

        self.decoder_pred_delta = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_theta = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_alpha = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_beta = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_gamma = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_upper = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        # --------------------------------------------------------------------------

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

        decoder_pos_embed_delta = get_1d_sincos_pos_embed(self.decoder_pos_embed_delta.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_delta.data.copy_(torch.from_numpy(decoder_pos_embed_delta).float().unsqueeze(0))
        decoder_pos_embed_theta = get_1d_sincos_pos_embed(self.decoder_pos_embed_theta.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_theta.data.copy_(torch.from_numpy(decoder_pos_embed_theta).float().unsqueeze(0))
        decoder_pos_embed_alpha = get_1d_sincos_pos_embed(self.decoder_pos_embed_alpha.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_alpha.data.copy_(torch.from_numpy(decoder_pos_embed_alpha).float().unsqueeze(0))
        decoder_pos_embed_beta = get_1d_sincos_pos_embed(self.decoder_pos_embed_beta.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_beta.data.copy_(torch.from_numpy(decoder_pos_embed_beta).float().unsqueeze(0))
        decoder_pos_embed_gamma = get_1d_sincos_pos_embed(self.decoder_pos_embed_gamma.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_gamma.data.copy_(torch.from_numpy(decoder_pos_embed_gamma).float().unsqueeze(0))
        decoder_pos_embed_upper = get_1d_sincos_pos_embed(self.decoder_pos_embed_upper.shape[-1], int(self.num_patches),
                                                    cls_token=True)
        self.decoder_pos_embed_upper.data.copy_(torch.from_numpy(decoder_pos_embed_upper).float().unsqueeze(0))


    def random_masking(self, x_delta,x_theta,x_alpha,x_beta,x_gamma,x_upper, mask_ratio):
        N, L, D = x_delta.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device= x_delta.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_delta_masked = torch.gather(x_delta, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_theta_masked = torch.gather(x_theta, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_alpha_masked = torch.gather(x_alpha, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_beta_masked = torch.gather(x_beta, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_gamma_masked = torch.gather(x_gamma, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_upper_masked = torch.gather(x_upper, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x_delta.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_delta_masked,x_theta_masked,x_alpha_masked,x_beta_masked,\
               x_gamma_masked,x_upper_masked, mask, ids_restore


    def patchify(self, series):
        x = series.reshape(shape=(series.shape[0], self.num_patches, -1))
        return x

    def forward_encoder(self, x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, mask_ratio):
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


        # masking
        x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, mask, ids_restore \
            = self.random_masking(x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper,mask_ratio)

        # append cls token
        cls_token_delta = self.cls_token_delta + self.pos_embed_delta[:, :1, :]
        cls_tokens_delta = cls_token_delta.expand(x_delta.shape[0], -1, -1)
        x_delta = torch.cat((cls_tokens_delta, x_delta), dim=1)

        cls_token_theta = self.cls_token_theta + self.pos_embed_theta[:, :1, :]
        cls_tokens_theta = cls_token_theta.expand(x_theta.shape[0], -1, -1)
        x_theta = torch.cat((cls_tokens_theta, x_theta), dim=1)

        cls_token_alpha = self.cls_token_alpha + self.pos_embed_alpha[:, :1, :]
        cls_tokens_alpha = cls_token_alpha.expand(x_alpha.shape[0], -1, -1)
        x_alpha = torch.cat((cls_tokens_alpha, x_alpha), dim=1)

        cls_token_beta = self.cls_token_beta + self.pos_embed_beta[:, :1, :]
        cls_tokens_beta = cls_token_beta.expand(x_beta.shape[0], -1, -1)
        x_beta = torch.cat((cls_tokens_beta, x_beta), dim=1)

        cls_token_gamma = self.cls_token_gamma + self.pos_embed_gamma[:, :1, :]
        cls_tokens_gamma = cls_token_gamma.expand(x_gamma.shape[0], -1, -1)
        x_gamma = torch.cat((cls_tokens_gamma, x_gamma), dim=1)

        cls_token_upper = self.cls_token_upper + self.pos_embed_upper[:, :1, :]
        cls_tokens_upper = cls_token_upper.expand(x_upper.shape[0], -1, -1)
        x_upper = torch.cat((cls_tokens_upper, x_upper), dim=1)

        # apply Transformer blocks
        x_delta = self.blocks_delta(x_delta)
        x_theta = self.blocks_theta(x_theta)
        x_alpha = self.blocks_alpha(x_alpha)
        x_beta = self.blocks_beta(x_beta)
        x_gamma = self.blocks_gamma(x_gamma)
        x_upper = self.blocks_upper(x_upper)

        return x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, mask, ids_restore

    def forward_decoder(self, x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, ids_restore):
        # embed tokens
        x_delta = self.decoder_embed_delta(x_delta)
        x_theta = self.decoder_embed_theta(x_theta)
        x_alpha = self.decoder_embed_alpha(x_alpha)
        x_beta = self.decoder_embed_beta(x_beta)
        x_gamma = self.decoder_embed_gamma(x_gamma)
        x_upper = self.decoder_embed_upper(x_upper)

        # append mask tokens to sequence
        mask_tokens_delta = self.mask_token_delta.repeat(x_delta.shape[0], ids_restore.shape[1] + 1 - x_delta.shape[1], 1)
        x_delta_ = torch.cat([x_delta[:, 1:, :], mask_tokens_delta], dim=1)  # no cls token
        x_delta_ = torch.gather(x_delta_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_delta.shape[2]))  # unshuffle
        x_delta = torch.cat([x_delta[:, :1, :], x_delta_], dim=1)  # append cls token

        # append mask tokens to sequence
        mask_tokens_theta = self.mask_token_theta.repeat(x_theta.shape[0], ids_restore.shape[1] + 1 - x_theta.shape[1], 1)
        x_theta_ = torch.cat([x_theta[:, 1:, :], mask_tokens_theta], dim=1)  # no cls token
        x_theta_ = torch.gather(x_theta_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_theta.shape[2]))  # unshuffle
        x_theta = torch.cat([x_theta[:, :1, :], x_theta_], dim=1)  # append cls token

        # append mask tokens to sequence
        mask_tokens_alpha = self.mask_token_alpha.repeat(x_alpha.shape[0], ids_restore.shape[1] + 1 - x_alpha.shape[1], 1)
        x_alpha_ = torch.cat([x_alpha[:, 1:, :], mask_tokens_alpha], dim=1)  # no cls token
        x_alpha_ = torch.gather(x_alpha_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_alpha.shape[2]))  # unshuffle
        x_alpha = torch.cat([x_alpha[:, :1, :], x_alpha_], dim=1)  # append cls token

        # append mask tokens to sequence
        mask_tokens_beta = self.mask_token_beta.repeat(x_beta.shape[0], ids_restore.shape[1] + 1 - x_beta.shape[1], 1)
        x_beta_ = torch.cat([x_beta[:, 1:, :], mask_tokens_beta], dim=1)  # no cls token
        x_beta_ = torch.gather(x_beta_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_beta.shape[2]))  # unshuffle
        x_beta = torch.cat([x_beta[:, :1, :], x_beta_], dim=1)  # append cls token

        # append mask tokens to sequence
        mask_tokens_gamma = self.mask_token_gamma.repeat(x_gamma.shape[0], ids_restore.shape[1] + 1 - x_gamma.shape[1], 1)
        x_gamma_ = torch.cat([x_gamma[:, 1:, :], mask_tokens_gamma], dim=1)  # no cls token
        x_gamma_ = torch.gather(x_gamma_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_gamma.shape[2]))  # unshuffle
        x_gamma = torch.cat([x_gamma[:, :1, :], x_gamma_], dim=1)  # append cls token

        # append mask tokens to sequence
        mask_tokens_upper = self.mask_token_upper.repeat(x_upper.shape[0], ids_restore.shape[1] + 1 - x_upper.shape[1], 1)
        x_upper_ = torch.cat([x_upper[:, 1:, :], mask_tokens_upper], dim=1)  # no cls token
        x_upper_ = torch.gather(x_upper_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_upper.shape[2]))  # unshuffle
        x_upper = torch.cat([x_upper[:, :1, :], x_upper_], dim=1)  # append cls token

        # # apply Transformer blocks
        x_delta = self.decoder_blocks_delta(x_delta)
        x_theta = self.decoder_blocks_theta(x_theta)
        x_alpha = self.decoder_blocks_alpha(x_alpha)
        x_beta = self.decoder_blocks_beta(x_beta)
        x_gamma = self.decoder_blocks_gamma(x_gamma)
        x_upper = self.decoder_blocks_upper(x_upper)


        # predictor projection
        x_delta = self.decoder_pred_delta(x_delta)
        x_theta = self.decoder_pred_theta(x_theta)
        x_alpha = self.decoder_pred_alpha(x_alpha)
        x_beta = self.decoder_pred_beta(x_beta)
        x_gamma = self.decoder_pred_gamma(x_gamma)
        x_upper = self.decoder_pred_upper(x_upper)


        # remove cls token
        x_delta = x_delta[:, 1:, :]
        x_theta = x_theta[:, 1:, :]
        x_alpha = x_alpha[:, 1:, :]
        x_beta = x_beta[:, 1:, :]
        x_gamma = x_gamma[:, 1:, :]
        x_upper = x_upper[:, 1:, :]

        x_combine = x_delta + x_theta + x_alpha + x_beta + x_gamma + x_upper

        return x_delta, x_theta, x_alpha, x_beta, \
               x_gamma, x_upper, x_combine

    def forward_loss(self, x, pred, mask):
        target = self.patchify(x)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, Data, delta,theta,alpha, beta,gamma,upper, mask_ratio):
        # mean = torch.mean(x, dim=-1)
        # mean = mean.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # std = torch.std(x, dim=-1)
        # std = std.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # x = (x - mean) / std

        x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, mask, ids_restore = self.forward_encoder(delta,theta,alpha, beta,gamma,upper, mask_ratio)
        x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, x_combine = self.forward_decoder(x_delta,x_theta,x_alpha, x_beta,x_gamma,x_upper, ids_restore)
        loss = self.forward_loss(Data, x_combine, mask)
        return loss, x_combine, mask