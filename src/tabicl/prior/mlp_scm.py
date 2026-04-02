
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
        self.graph_sparsity = float(kwargs.get("graph_sparsity", 0.2))
        if not (0.0 <= self.graph_sparsity <= 1.0):
            raise ValueError(f"graph_sparsity must be in [0, 1], got {self.graph_sparsity}")

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
        X = self.apply_graph_sparsity(X)

        # 最终安全性检查
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        
        # 限制 X 到一个 Transformer 易于处理的范围
        X = torch.clamp(X, -15.0, 15.0)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def apply_graph_sparsity(self, X: torch.Tensor) -> torch.Tensor:
        """根据 graph_sparsity 对部分特征列做行置换，削弱其与 y 的直接对应关系。"""
        if self.graph_sparsity <= 0.0 or X.shape[1] == 0:
            return X

        mask = torch.rand((X.shape[1],), device=X.device) < self.graph_sparsity
        if not bool(mask.any()):
            return X

        X = X.clone()
        perm = torch.randperm(X.shape[0], device=X.device)
        X[:, mask] = X[perm][:, mask]
        return X

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
