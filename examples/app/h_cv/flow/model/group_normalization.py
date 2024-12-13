import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def group_norm(
    input,
    group,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
):
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError(
            "Expected running_mean and running_var to be not None when use_input_stats=False"
        )

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(
        input,
        group,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        use_input_stats=None,
        momentum=None,
        eps=None,
    ):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        # norm_shape = [1, b * c / group, group]
        # print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(
            1, int(b * c / group), group, *input.size()[2:]
        )

        out = F.batch_norm(
            input_reshaped,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=use_input_stats,
            momentum=momentum,
            eps=eps,
        )

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(
                running_mean.view(b, int(c / group)).mean(0, keepdim=False)
            )
        if running_var is not None:
            running_var_orig.copy_(
                running_var.view(b, int(c / group)).mean(0, keepdim=False)
            )

        return out.view(b, c, *input.size()[2:])

    return _instance_norm(
        input,
        group,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )


class _GroupNorm(_BatchNorm):
    def __init__(
        self,
        num_features,
        num_groups=1,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(
            int(num_features / num_groups), eps, momentum, affine, track_running_stats
        )

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input,
            self.num_groups,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class GroupNorm2d(_GroupNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class GroupNorm3d(_GroupNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))