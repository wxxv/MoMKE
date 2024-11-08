import torch
from torch import nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_softmoe(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Transformer_a = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, cross_modality='atv', mask_modality=None, mask=None):
        # x: [B, s, C]
        B, s, C = x.shape
        if cross_modality == 'a':
            x_a_mlp = self.Transformer_a(x, mask_modality, mask)
            return x_a_mlp
        if cross_modality == 't':
            x_t_mlp = self.Transformer_t(x, mask_modality, mask)
            return x_t_mlp
        if cross_modality == 'v':
            x_v_mlp = self.Transformer_v(x, mask_modality, mask)
            return x_v_mlp



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )


    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        if mask is not None:
            mask = mask.bool()
            mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
            mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)

        return x_out


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.drop = drop

        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio,)
                for i in range(depth)
            ]
        )

    def forward(self, x, first_stage, mask=None, modality=None):
        if first_stage:
            for layer_idx, block in enumerate(self.blocks):
                x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
            return x
        else:
            x_cross_a, x_cross_t, x_cross_v = torch.clone(x), torch.clone(x), torch.clone(x)
            for layer_idx, block in enumerate(self.blocks):
                x_cross_a = x_cross_a + block(x_cross_a, cross_modality='a', mask_modality=modality, mask=mask)
                x_cross_t = x_cross_t + block(x_cross_t, cross_modality='t', mask_modality=modality, mask=mask)
                x_cross_v = x_cross_v + block(x_cross_v, cross_modality='v', mask_modality=modality, mask=mask)
            return torch.cat([x_cross_a, x_cross_t, x_cross_v], dim=-1)
