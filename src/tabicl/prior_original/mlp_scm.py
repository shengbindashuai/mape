# from __future__ import annotations

# import math
# import random
# from typing import Dict, Any

# import torch
# from torch import nn

# from .utils import GaussianNoise, XSampler


# class MLPSCM(nn.Module):
#     """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).

#     Parameters
#     ----------
#     seq_len : int, default=1024
#         The number of samples (rows) to generate for the dataset.

#     num_features : int, default=100
#         The number of features.

#     num_outputs : int, default=1
#         The number of outputs.

#     is_causal : bool, default=True
#         - If `True`, simulates a causal graph: `X` and `y` are sampled from the
#           intermediate hidden states of the MLP transformation applied to initial causes.
#           The `num_causes` parameter controls the number of initial root variables.
#         - If `False`, simulates a direct predictive mapping: Initial causes are used
#           directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
#           is effectively ignored and set equal to `num_features`.

#     num_causes : int, default=10
#         The number of initial root 'cause' variables sampled by `XSampler`.
#         Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
#         set to `num_features`.

#     y_is_effect : bool, default=True
#         Specifies how the target `y` is selected when `is_causal=True`.
#         - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
#           representing terminal effects in the causal chain.
#         - If `False`, `y` is sampled from the earlier intermediate outputs (after
#           permutation), representing variables closer to the initial causes.

#     in_clique : bool, default=False
#         Controls how features `X` and targets `y` are sampled from the flattened
#         intermediate MLP outputs when `is_causal=True`.
#         - If `True`, `X` and `y` are selected from a contiguous block of the
#           intermediate outputs, potentially creating denser dependencies among them.
#         - If `False`, `X` and `y` indices are chosen randomly and independently
#           from all available intermediate outputs.

#     sort_features : bool, default=True
#         Determines whether to sort the features based on their original indices from
#         the intermediate MLP outputs. Only relevant when `is_causal=True`.

#     num_layers : int, default=10
#         The total number of layers in the MLP transformation network. Must be >= 2.
#         Includes the initial linear layer and subsequent blocks of
#         (Activation -> Linear -> Noise).

#     hidden_dim : int, default=20
#         The dimensionality of the hidden representations within the MLP layers.
#         If `is_causal=True`, this is automatically increased if it's smaller than
#         `num_outputs + 2 * num_features` to ensure enough intermediate variables
#         are generated for sampling `X` and `y`.

#     mlp_activations : default=nn.Tanh
#         The activation function to be used after each linear transformation
#         in the MLP layers (except the first).

#     init_std : float, default=1.0
#         The standard deviation of the normal distribution used for initializing
#         the weights of the MLP's linear layers.

#     block_wise_dropout : bool, default=True
#         Specifies the weight initialization strategy.
#         - If `True`, uses a 'block-wise dropout' initialization where only random
#           blocks within the weight matrix are initialized with values drawn from
#           a normal distribution (scaled by `init_std` and potentially dropout),
#           while the rest are zero. This encourages sparsity.
#         - If `False`, uses standard normal initialization for all weights, followed
#           by applying dropout mask based on `mlp_dropout_prob`.

#     mlp_dropout_prob : float, default=0.1
#         The dropout probability applied to weights during *standard* initialization
#         (i.e., when `block_wise_dropout=False`). Ignored if
#         `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

#     scale_init_std_by_dropout : bool, default=True
#         Whether to scale the `init_std` during weight initialization to compensate
#         for the variance reduction caused by dropout. If `True`, `init_std` is
#         divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
#         initialization method.

#     sampling : str, default="normal"
#         The method used by `XSampler` to generate the initial 'cause' variables.
#         Options:
#         - "normal": Standard normal distribution (potentially with pre-sampled stats).
#         - "uniform": Uniform distribution between 0 and 1.
#         - "mixed": A random combination of normal, multinomial (categorical),
#           Zipf (power-law), and uniform distributions across different cause variables.

#     pre_sample_cause_stats : bool, default=False
#         If `True` and `sampling="normal"`, the mean and standard deviation for
#         each initial cause variable are pre-sampled. Passed to `XSampler`.

#     noise_std : float, default=0.01
#         The base standard deviation for the Gaussian noise added after each MLP
#         layer's linear transformation (except the first layer).

#     pre_sample_noise_std : bool, default=False
#         Controls how the standard deviation for the `GaussianNoise` layers is determined.

#     device : str, default="cpu"
#         The computing device ('cpu' or 'cuda') where tensors will be allocated.

#     **kwargs : dict
#         Unused hyperparameters passed from parent configurations.
#     """

#     def __init__(
#         self,
#         seq_len: int = 1024,
#         num_features: int = 100,
#         num_outputs: int = 1,
#         is_causal: bool = True,
#         num_causes: int = 10,
#         y_is_effect: bool = True,
#         in_clique: bool = False,
#         sort_features: bool = True,
#         num_layers: int = 10,
#         hidden_dim: int = 20,
#         mlp_activations: Any = nn.Tanh,
#         init_std: float = 1.0,
#         block_wise_dropout: bool = True,
#         mlp_dropout_prob: float = 0.1,
#         scale_init_std_by_dropout: bool = True,
#         sampling: str = "normal",
#         pre_sample_cause_stats: bool = False,
#         noise_std: float = 0.01,
#         pre_sample_noise_std: bool = False,
#         device: str = "cpu",
#         **kwargs: Dict[str, Any],
#     ):
#         super(MLPSCM, self).__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.num_outputs = num_outputs
#         self.is_causal = is_causal
#         self.num_causes = num_causes
#         self.y_is_effect = y_is_effect
#         self.in_clique = in_clique
#         self.sort_features = sort_features

#         assert num_layers >= 2, "Number of layers must be at least 2."
#         self.num_layers = num_layers

#         self.hidden_dim = hidden_dim
#         self.mlp_activations = mlp_activations
#         self.init_std = init_std
#         self.block_wise_dropout = block_wise_dropout
#         self.mlp_dropout_prob = mlp_dropout_prob
#         self.scale_init_std_by_dropout = scale_init_std_by_dropout
#         self.sampling = sampling
#         self.pre_sample_cause_stats = pre_sample_cause_stats
#         self.noise_std = noise_std
#         self.pre_sample_noise_std = pre_sample_noise_std
#         self.device = device

#         if self.is_causal:
#             # Ensure enough intermediate variables for sampling X and y
#             self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
#         else:
#             # In non-causal mode, features are the causes
#             self.num_causes = self.num_features

#         # Define the input sampler
#         self.xsampler = XSampler(
#             self.seq_len,
#             self.num_causes,
#             pre_stats=self.pre_sample_cause_stats,
#             sampling=self.sampling,
#             device=self.device,
#         )

#         # Build layers
#         layers = [nn.Linear(self.num_causes, self.hidden_dim)]
#         for _ in range(self.num_layers - 1):
#             layers.append(self.generate_layer_modules())
#         if not self.is_causal:
#             layers.append(self.generate_layer_modules(is_output_layer=True))
#         self.layers = nn.Sequential(*layers).to(device)

#         # Initialize layers
#         self.initialize_parameters()

#     def generate_layer_modules(self, is_output_layer=False):
#         """Generates a layer module with activation, linear transformation, and noise."""
#         out_dim = self.num_outputs if is_output_layer else self.hidden_dim
#         activation = self.mlp_activations()
#         linear_layer = nn.Linear(self.hidden_dim, out_dim)

#         if self.pre_sample_noise_std:
#             noise_std = torch.abs(
#                 torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
#             )
#         else:
#             noise_std = self.noise_std
#         noise_layer = GaussianNoise(noise_std)

#         return nn.Sequential(activation, linear_layer, noise_layer)

#     def initialize_parameters(self):
#         """Initializes parameters using block-wise dropout or normal initialization."""
#         for i, (_, param) in enumerate(self.layers.named_parameters()):
#             if self.block_wise_dropout and param.dim() == 2:
#                 self.initialize_with_block_dropout(param, i)
#             else:
#                 self.initialize_normally(param, i)

#     def initialize_with_block_dropout(self, param, index):
#         """Initializes parameters using block-wise dropout."""
#         nn.init.zeros_(param)
#         n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
#         block_size = [dim // n_blocks for dim in param.shape]
#         keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
#         for block in range(n_blocks):
#             block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
#             nn.init.normal_(
#                 param[block_slice], std=self.init_std / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1)
#             )

#     def initialize_normally(self, param, index):
#         """Initializes parameters using normal distribution."""
#         if param.dim() == 2:  # Applies only to weights, not biases
#             dropout_prob = self.mlp_dropout_prob if index > 0 else 0  # No dropout for the first layer's weights
#             dropout_prob = min(dropout_prob, 0.99)
#             std = self.init_std / ((1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1)
#             nn.init.normal_(param, std=std)
#             param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

#     def forward(self):
#         """Generates synthetic data by sampling input features and applying MLP transformations."""
#         causes = self.xsampler.sample()  # (seq_len, num_causes)

#         # Generate outputs through MLP layers
#         outputs = [causes]
#         for layer in self.layers:
#             outputs.append(layer(outputs[-1]))
#         outputs = outputs[2:]  # Start from 2 because the first layer is only linear without activation

#         # Handle outputs based on causality
#         X, y = self.handle_outputs(causes, outputs)

#         # Check for NaNs and handle them by setting to default values
#         if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
#             X[:] = 0.0
#             y[:] = -100.0

#         if self.num_outputs == 1:
#             y = y.squeeze(-1)

#         return X, y

#     def handle_outputs(self, causes, outputs):
#         """
#         Handles outputs based on whether causal or not.

#         If causal, sample inputs and target from the graph.
#         If not causal, directly use causes as inputs and last output as target.

#         Parameters
#         ----------
#         causes : torch.Tensor
#             Causes of shape (seq_len, num_causes)

#         outputs : list of torch.Tensor
#             List of output tensors from MLP layers

#         Returns
#         -------
#         X : torch.Tensor
#             Input features (seq_len, num_features)

#         y : torch.Tensor
#             Target (seq_len, num_outputs)
#         """
#         if self.is_causal:
#             outputs_flat = torch.cat(outputs, dim=-1)
#             if self.in_clique:
#                 # When in_clique=True, features and targets are sampled as a block, ensuring that
#                 # selected variables may share dense dependencies.
#                 start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
#                 random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
#             else:
#                 # Otherwise, features and targets are randomly and independently selected
#                 random_perm = torch.randperm(outputs_flat.shape[-1] - 1, device=self.device)

#             indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
#             if self.y_is_effect:
#                 # If targets are effects, take last output dims
#                 indices_y = list(range(-self.num_outputs, 0))
#             else:
#                 # Otherwise, take from the beginning of the permuted list
#                 indices_y = random_perm[: self.num_outputs]

#             if self.sort_features:
#                 indices_X, _ = torch.sort(indices_X)

#             # Select input features and targets from outputs
#             X = outputs_flat[:, indices_X]
#             y = outputs_flat[:, indices_y]
#         else:
#             # In non-causal mode, use original causes and last layer output
#             X = causes
#             y = outputs[-1]

#         return X, y


# from __future__ import annotations

# import math
# import random
# from typing import Dict, Any

# import torch
# from torch import nn

# from .utils import GaussianNoise, XSampler


# class MLPSCM(nn.Module):
#     def __init__(
#         self,
#         seq_len: int = 1024,
#         num_features: int = 100,
#         num_outputs: int = 1,
#         is_causal: bool = True,
#         num_causes: int = 10,
#         y_is_effect: bool = True,
#         in_clique: bool = False,
#         sort_features: bool = True,
#         num_layers: int = 10,
#         hidden_dim: int = 20,
#         mlp_activations: Any = nn.Tanh, # Tanh 比 ReLU 更能抑制梯度爆炸
#         init_std: float = 0.5,           # 降低默认初始方差
#         block_wise_dropout: bool = True,
#         mlp_dropout_prob: float = 0.1,
#         scale_init_std_by_dropout: bool = True,
#         sampling: str = "normal",
#         pre_sample_cause_stats: bool = False,
#         noise_std: float = 0.01,
#         pre_sample_noise_std: bool = False,
#         device: str = "cpu",
#         **kwargs: Dict[str, Any],
#     ):
#         super(MLPSCM, self).__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.num_outputs = num_outputs
#         self.is_causal = is_causal
#         self.num_causes = num_causes
#         self.y_is_effect = y_is_effect
#         self.in_clique = in_clique
#         self.sort_features = sort_features

#         assert num_layers >= 2, "Number of layers must be at least 2."
#         self.num_layers = num_layers

#         self.hidden_dim = hidden_dim
#         self.mlp_activations = mlp_activations
#         self.init_std = init_std
#         self.block_wise_dropout = block_wise_dropout
#         self.mlp_dropout_prob = mlp_dropout_prob
#         self.scale_init_std_by_dropout = scale_init_std_by_dropout
#         self.sampling = sampling
#         self.pre_sample_cause_stats = pre_sample_cause_stats
#         self.noise_std = noise_std
#         self.pre_sample_noise_std = pre_sample_noise_std
#         self.device = device

#         if self.is_causal:
#             self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
#         else:
#             self.num_causes = self.num_features

#         self.xsampler = XSampler(
#             self.seq_len,
#             self.num_causes,
#             pre_stats=self.pre_sample_cause_stats,
#             sampling=self.sampling,
#             device=self.device,
#         )

#         # 构建层
#         layers = [nn.Linear(self.num_causes, self.hidden_dim)]
#         for _ in range(self.num_layers - 1):
#             layers.append(self.generate_layer_modules())
#         if not self.is_causal:
#             layers.append(self.generate_layer_modules(is_output_layer=True))
#         self.layers = nn.Sequential(*layers).to(device)

#         self.initialize_parameters()

#     def generate_layer_modules(self, is_output_layer=False):
#         out_dim = self.num_outputs if is_output_layer else self.hidden_dim
#         activation = self.mlp_activations()
#         linear_layer = nn.Linear(self.hidden_dim, out_dim)

#         if self.pre_sample_noise_std:
#             noise_std = torch.abs(
#                 torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
#             )
#         else:
#             noise_std = self.noise_std
#         noise_layer = GaussianNoise(noise_std)

#         return nn.Sequential(activation, linear_layer, noise_layer)

#     def initialize_parameters(self):
#         for i, (name, param) in enumerate(self.layers.named_parameters()):
#             if "weight" in name and param.dim() == 2:
#                 if self.block_wise_dropout:
#                     self.initialize_with_block_dropout(param, i)
#                 else:
#                     self.initialize_normally(param, i)
#             elif "bias" in name:
#                 nn.init.zeros_(param)

#     def initialize_with_block_dropout(self, param, index):
#         nn.init.zeros_(param)
#         n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
#         block_size = [dim // n_blocks for dim in param.shape]
#         keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
#         for block in range(n_blocks):
#             block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
#             # 引入额外的 0.1 系数防止初始值过大
#             std = (self.init_std * 0.1) / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1)
#             nn.init.normal_(param[block_slice], std=std)

#     def initialize_normally(self, param, index):
#         dropout_prob = self.mlp_dropout_prob if index > 0 else 0
#         dropout_prob = min(dropout_prob, 0.99)
#         # 引入 0.1 系数缩小权重
#         std = (self.init_std * 0.1) / ((1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1)
#         nn.init.normal_(param, std=std)
#         param.data *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

#     def forward(self):
#         causes = self.xsampler.sample() 
#         # 截断初始输入
#         causes = torch.clamp(causes, -10.0, 10.0)

#         outputs = [causes]
#         for layer in self.layers:
#             next_val = layer(outputs[-1])
            
#             # --- 关键修改：逐层截断和 NaN 处理 ---
#             if torch.isnan(next_val).any() or torch.isinf(next_val).any():
#                 next_val = torch.nan_to_num(next_val, nan=0.0, posinf=10.0, neginf=-10.0)
            
#             # 每一层输出后进行截断，防止数值爆炸级联
#             next_val = torch.clamp(next_val, -10.0, 10.0)
#             outputs.append(next_val)

#         outputs = outputs[2:] 

#         X, y = self.handle_outputs(causes, outputs)

#         # --- 关键修改：分类任务下特征 X 的最终清洗 ---
#         if torch.any(torch.isnan(X)):
#             X = torch.nan_to_num(X, nan=0.0)
        
#         # 将特征 X 限制在合理范围，并做简单的鲁棒缩放
#         # 这样进入 T5 文本编码器的数字不会有极长的情况
#         X = torch.clamp(X, -10.0, 10.0)

#         # 分类任务 y 必须是长整型，这里假设后续会有转化为 Long 的操作
#         if torch.any(torch.isnan(y)):
#             # 分类任务 NaN 设为一个无效类别或 0
#             y = torch.nan_to_num(y, nan=0.0)

#         if self.num_outputs == 1:
#             y = y.squeeze(-1)

#         return X, y

#     def handle_outputs(self, causes, outputs):
#         if self.is_causal:
#             # 拼接所有中间层输出
#             outputs_flat = torch.cat(outputs, dim=-1)
            
#             # 增加对 outputs_flat 的安全性检查
#             outputs_flat = torch.clamp(outputs_flat, -15.0, 15.0)

#             if self.in_clique:
#                 max_start = outputs_flat.shape[-1] - self.num_outputs - self.num_features
#                 start = random.randint(0, max(0, max_start))
#                 random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
#             else:
#                 random_perm = torch.randperm(outputs_flat.shape[-1], device=self.device)

#             # 确保索引不越界
#             indices_X = random_perm[:self.num_features]
            
#             if self.y_is_effect:
#                 # 效果 y 通常选在较深层（列表末尾对应的索引）
#                 indices_y = torch.arange(outputs_flat.shape[-1] - self.num_outputs, outputs_flat.shape[-1], device=self.device)
#             else:
#                 indices_y = random_perm[self.num_features : self.num_features + self.num_outputs]

#             if self.sort_features:
#                 indices_X, _ = torch.sort(indices_X)

#             X = outputs_flat[:, indices_X]
#             y = outputs_flat[:, indices_y]
#         else:
#             X = causes
#             y = outputs[-1]

#         return X, y


from __future__ import annotations

import math
import random
from typing import Dict, Any

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .utils import GaussianNoise, XSampler


class MLPSCM(nn.Module):
    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 0.1,  # 降低初始化强度
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super(MLPSCM, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.device = device

        assert num_layers >= 2, "Number of layers must be at least 2."
        self.num_layers = num_layers

        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std

        if self.is_causal:
            self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
        else:
            self.num_causes = self.num_features

        # 定义输入采样器
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # 构建层 (使用 ModuleList 方便 forward 里的残差操作)
        self.layers = self._build_layers()
        
        # 初始化
        self.initialize_parameters()

    def _build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        # 第一层：线性映射
        layers.append(nn.Linear(self.num_causes, self.hidden_dim))
        
        # 中间层：Activation -> Linear (Spectral Norm) -> Noise
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
            
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        return layers

    def generate_layer_modules(self, is_output_layer=False):
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        
        # 核心改动：应用谱归一化限制层权重增益
        linear_layer = spectral_norm(nn.Linear(self.hidden_dim, out_dim))

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() == 2:
                # 使用较小的标准差初始化，防止信号过强
                if self.block_wise_dropout:
                    self.initialize_with_block_dropout(param)
                else:
                    nn.init.normal_(param, std=self.init_std * 0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initialize_with_block_dropout(self, param):
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [max(1, dim // n_blocks) for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
            std = (self.init_std * 0.05) / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1)
            nn.init.normal_(param[block_slice], std=std)

    def forward(self):
        causes = self.xsampler.sample()  # (seq_len, num_causes)
        # 限制初始范围
        causes = torch.clamp(causes, -5.0, 5.0)

        outputs = []
        current_state = None
        res_scale = 0.2  # 残差步长，稳定数值流

        for i, layer in enumerate(self.layers):
            if i == 0:
                current_state = layer(causes)
            else:
                z = layer(current_state)
                # 执行残差连接：新状态 = 旧状态 + 缩放后的变化
                if z.shape == current_state.shape:
                    current_state = current_state + res_scale * z
                else:
                    current_state = z
            
            # 每一层后的数值稳定化
            if torch.isnan(current_state).any():
                current_state = torch.nan_to_num(current_state, 0.0)
            
            # 软性限幅：防止数值随层数加深而累积爆炸
            current_state = torch.clamp(current_state, -10.0, 10.0)
            outputs.append(current_state)

        # 这里的 [1:] 是为了跳过第一层单纯线性变换的结果，增加非线性复杂度
        X, y = self.handle_outputs(causes, outputs[1:])

        # 最终安全性检查
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        
        # 限制 X 到一个 Transformer 易于处理的范围
        X = torch.clamp(X, -15.0, 15.0)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def handle_outputs(self, causes, outputs):
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            
            # 局部归一化：确保所有拼接后的特征在同一量级
            eps = 1e-6
            outputs_flat = (outputs_flat - outputs_flat.mean(0)) / (outputs_flat.std(0) + eps)

            if self.in_clique:
                max_range = outputs_flat.shape[-1] - self.num_outputs - self.num_features
                start = random.randint(0, max(0, max_range))
                random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
            else:
                random_perm = torch.randperm(outputs_flat.shape[-1], device=self.device)

            indices_X = random_perm[:self.num_features]
            if self.y_is_effect:
                # 取最后的节点作为 y (结果)
                indices_y = torch.arange(outputs_flat.shape[-1] - self.num_outputs, outputs_flat.shape[-1], device=self.device)
            else:
                indices_y = random_perm[self.num_features : self.num_features + self.num_outputs]

            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)

            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            X = causes
            y = outputs[-1]

        return X, y