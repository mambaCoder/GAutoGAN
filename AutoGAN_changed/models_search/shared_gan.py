# -*- coding: utf-8 -*-
# @Date    : 2019-08-15
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
import torch.nn as nn

from models_search.building_blocks_search import *


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)  # 变成4*4
        self.cell1 = Cell(args.gf_dim, args.gf_dim, num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        arch_stage1 = arch_id[:4]
        self.cell1.set_arch(conv_id=arch_stage1[0], norm_id=arch_stage1[1], up_id=arch_stage1[2],
                            short_cut_id=arch_stage1[3], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[4:9]
            self.cell2.set_arch(conv_id=arch_stage2[0], norm_id=arch_stage2[1], up_id=arch_stage2[2],
                                short_cut_id=arch_stage2[3], skip_ins=arch_stage2[4])

        if cur_stage == 2:
            arch_stage3 = arch_id[12:]
            self.cell3.set_arch(conv_id=arch_stage3[0], norm_id=arch_stage3[1], up_id=arch_stage3[2],
                                short_cut_id=arch_stage3[3], skip_ins=arch_stage3[4])

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        if self.cur_stage == 0:
            return self.to_rgb(h1)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        if self.cur_stage == 1:
            return self.to_rgb(h2)
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        if self.cur_stage == 2:
            return self.to_rgb(h3)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.cell1 = OptimizedDisBlock(args, 3, self.ch)
        self.cell2 = DisCell(args, self.ch, self.ch, activation=activation, downsample=True)
        self.cell3 = DisCell(args, self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)
        self.cur_stage = 0

    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        # arch_stage1 = arch_id[:6]
        # self.cell1.set_arch(disconv_id=arch_stage1[4], norm_id=arch_stage1[5], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[9:12]
            self.cell2.set_arch(disconv_id=arch_stage2[0], norm_id=arch_stage2[1], sc_id=arch_stage2[2])

        if cur_stage == 2:
            arch_stage3 = arch_id[17:]
            self.cell3.set_arch(disconv_id=arch_stage3[0], norm_id=arch_stage3[1], sc_id=arch_stage2[2])

    def forward(self, x):
        h = x
        layers = [self.cell1, self.cell2, self.cell3]
        variable_model = nn.Sequential(*layers[:(self.cur_stage + 1)])
        h = variable_model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output

