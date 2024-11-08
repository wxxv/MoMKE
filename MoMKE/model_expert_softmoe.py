import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention_softmoe import *


class MoMKE(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(MoMKE, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        self.block = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=depth,
                )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)

        self.router_a = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_t = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_v = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])
        weight_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # weight
        weight_a, weight_t, weight_v = self.router_a(proj_a), self.router_t(proj_t), self.router_v(proj_v)
        weight_a = torch.softmax(weight_a, dim=-1)
        weight_t = torch.softmax(weight_t, dim=-1)
        weight_v = torch.softmax(weight_v, dim=-1)
        weight_save.append(np.array([weight_a.cpu().detach().numpy(), weight_t.cpu().detach().numpy(), weight_v.cpu().detach().numpy()]))
        weight_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        # --> [batch, 3*seqlen, dim]
        x_a = self.block(proj_a, first_stage, attn_mask, 'a')
        x_t = self.block(proj_t, first_stage, attn_mask, 't')
        x_v = self.block(proj_v, first_stage, attn_mask, 'v')
        if first_stage:
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            # meaningless
            out_a, out_t, out_v = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))

            x_unweighted_a = x_a.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_t = x_t.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_v = x_v.reshape(B, seq_len, 3, self.D_e)
            x_out_a = torch.sum(weight_a * x_unweighted_a, dim=2)
            x_out_t = torch.sum(weight_t * x_unweighted_t, dim=2)
            x_out_v = torch.sum(weight_v * x_unweighted_v, dim=2)
            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)

        x[attn_mask == 0] = 0

        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)

        return hidden, out, out_a, out_t, out_v, np.array(weight_save)


if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = MoMKE(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
