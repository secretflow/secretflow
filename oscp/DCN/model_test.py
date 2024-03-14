##测试 DeepCross

import torch
from model import DeepCross

x_num = torch.randn(2, 3)
x_cat = torch.randint(0, 2, (2, 3))

dcn_vec = DeepCross(
    d_numerical=3,
    categories=[563, 16, 4, 10, 70, 41283, 16, 4731, 26, 240748, 321439],
    d_embed_max=4,
    n_cross=2,
    cross_type="vector",
    mlp_layers=[20, 20],
    mlp_dropout=0.25,
    stacked=False,
    n_classes=1,
)

dcn_matrix = DeepCross(
    d_numerical=3,
    categories=[4, 3, 2],
    d_embed_max=4,
    n_cross=2,
    cross_type="matrix",
    mlp_layers=[20, 20],
    mlp_dropout=0.25,
    stacked=True,
    n_classes=1,
)

dcn_mix = DeepCross(
    d_numerical=3,
    categories=[4, 3, 2],
    d_embed_max=4,
    n_cross=2,
    cross_type="mix",
    low_rank=32,
    n_experts=4,
    mlp_layers=[20, 20],
    mlp_dropout=0.25,
    stacked=False,
    n_classes=1,
)


print(dcn_vec((x_num, x_cat)))
print(dcn_matrix((x_num, x_cat)))
print(dcn_mix((x_num, x_cat)))
