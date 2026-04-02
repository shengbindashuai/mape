from __future__ import annotations

import math
import random
from typing import Dict, Any, List, Tuple

import torch
from torch import nn
import networkx as nx

from .utils import GaussianNoise, XSampler



class MLPSCM(nn.Module):
    """
    用 NetworkX 构建的结构因果模型（SCM）来生成表格数据。

    关键思路
    - 随机生成一个 DAG：节点 0..num_nodes-1，其中前 num_causes 个为根因（无父节点）；
    - 根因由 XSampler 直接采样；
    - 其余节点按拓扑序：z_i = f( sum_j w_ij * z_j ) + eps_i ，f 随机选自 {identity, tanh, relu, sigmoid}；
    - 最终从 DAG 的中间/末端节点里选择特征 X 与目标 y。

    参数基本兼容 MLPSCM 的用法：
    - seq_len, num_features, num_outputs, num_causes, y_is_effect, in_clique, sort_features,
      noise_std, pre_sample_noise_std, sampling, pre_sample_cause_stats, device, init_std 等。
    - 另外新增：num_nodes, edge_prob 控制图规模与稀疏度。

    注意：本类不训练权重，权重只用于一次性采样数据。
    """

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
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        # 图结构相关
        num_nodes: int | None = None,   # 若为 None，自动设为 num_causes + 2*num_features + num_outputs
        edge_prob: float = 0.3,        # 边存在概率（越大图越密）
        # 生成函数与噪声
        init_std: float = 1.0,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_causes = num_causes
        # self.y_is_effect = y_is_effect
        self.y_is_effect = random.random() < 0.7
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.init_std = init_std
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.device = device

        # 图规模：保证足够多的中间变量可以抽取 X 和 y
        if num_nodes is None:
            num_nodes = max(num_causes + 2 * num_features + num_outputs, num_causes + 8)
        assert num_nodes >= num_causes + num_features + num_outputs, \
            "num_nodes 至少需要覆盖根因与 X/y 的数量。"
        self.num_nodes = num_nodes
        # self.edge_prob = edge_prob
        self.edge_prob = random.uniform(0.1, 0.5)
        # y_is_effect = 

        # 输入采样器（只用于根因）
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # 构图与参数（父->子权重；每个非根节点一个非线性）
        self.G = self._build_random_dag(self.num_nodes, self.num_causes, self.edge_prob)
        self.topo_order = list(nx.topological_sort(self.G))
        self.activations = self._sample_node_activations(self.num_nodes, self.num_causes)
        self.weights = self._init_edge_weights(self.G, self.init_std)  # dict[(u,v)] -> torch.Tensor([w])

        # 预采样每个节点的噪声强度（若启用）
        if self.pre_sample_noise_std:
            self.node_noise_std = {
                i: torch.abs(
                    torch.normal(
                        mean=torch.tensor(0.0, device=self.device),
                        std=float(self.noise_std)
                    )
                ).item()
                for i in range(self.num_nodes)
            }
        else:
            self.node_noise_std = {i: self.noise_std for i in range(self.num_nodes)}

    # -------------------------
    # 构图与初始化
    # -------------------------
    def _build_random_dag(self, N: int, num_roots: int, p: float) -> nx.DiGraph:
        """
        保证 0..num_roots-1 为根（无父），其余节点允许从任意更小索引连边，形成 DAG。
        为避免孤立子图，会确保每个非根至少有一个父节点（若无则强制从前面随机补一条）。
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(N))

        # 候选边只允许 i -> j 且 i < j（天然无环）
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < p:
                    G.add_edge(i, j)

        # 禁止根节点有父；若出现，移除进入根的边
        for r in range(num_roots):
            in_edges = list(G.in_edges(r))
            G.remove_edges_from(in_edges)

        # 确保每个非根至少有一个父
        for j in range(num_roots, N):
            if G.in_degree(j) == 0:
                # 从 [0, j-1] 中随机挑一个父
                parent = random.randint(0, j - 1)
                # 避免把根的“无父”打破没有问题；根允许出边
                G.add_edge(parent, j)

        return G

    def _sample_node_activations(self, N: int, num_roots: int):
        """
        为每个节点分配一个简单非线性（根节点用恒等，因为直接是观测到的 exogenous）
        """
        choices = [
            nn.Identity(),
            nn.Tanh(),
            nn.ReLU(),
            nn.Sigmoid(),
        ]
        acts = {}
        for i in range(N):
            if i < num_roots:
                acts[i] = nn.Identity()
            else:
                acts[i] = random.choice(choices)
        return acts

    def _init_edge_weights(self, G: nx.DiGraph, std: float):
        """
        每条边一个标量权重 w_ij ~ N(0, std^2)
        """
        w = {}
        for (u, v) in G.edges():
            w[(u, v)] = torch.normal(
                mean=torch.tensor(0.0, device=self.device),
                std=float(std)
            )
        return w

    # -------------------------
    # 前向生成
    # -------------------------
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
        X: (seq_len, num_features)
        y: (seq_len, num_outputs) / 若 num_outputs==1，最后会 squeeze 成 (seq_len,)
        """
        # 容器：每个节点的取值（seq_len, 1）
        node_values: Dict[int, torch.Tensor] = {}

        # 先生成根因
        causes = self.xsampler.sample()  # (seq_len, num_causes)
        for i in range(self.num_causes):
            node_values[i] = causes[:, i : i + 1]  # 保持2D列向量

        # 按拓扑序生成其余节点
        for node in self.topo_order:
            if node < self.num_causes:
                continue
            parents = list(self.G.predecessors(node))
            if len(parents) == 0:
                # 理论上不会发生（构图时保证了至少一个父）
                z = torch.zeros(self.seq_len, 1, device=self.device)
            else:
                # 线性聚合 + 噪声
                agg = torch.zeros(self.seq_len, 1, device=self.device)
                for p in parents:
                    w = self.weights[(p, node)]
                    agg = agg + node_values[p] * w
                eps_std = float(self.node_noise_std[node])
                eps = torch.normal(
                    mean=torch.zeros(self.seq_len, 1, device=self.device),
                    std=eps_std
                )
                z = agg + eps

            # 非线性
            act = self.activations[node]
            node_values[node] = act(z)

        # 将所有节点拼成矩阵 (seq_len, num_nodes)
        all_nodes = torch.cat([node_values[i] for i in range(self.num_nodes)], dim=1)

        # 选择 X 与 y 的索引
        X, y = self._select_X_y(all_nodes)

        # NaN 保护
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    # -------------------------
    # X/y 抽取策略
    # -------------------------
    def _select_X_y(self, all_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        all_nodes: (seq_len, num_nodes)
        1) 若 y_is_effect=True，优先从出度为 0 的“末端”节点取 y；否则从靠近根的节点取 y；
        2) X 从除去 y 的节点里抽取 num_features 个（in_clique=True 时尽量连成团或拓扑相邻）；
        """
        # 候选 y
        if self.y_is_effect:
            sinks = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
            y_candidates = sinks if len(sinks) >= self.num_outputs else list(range(self.num_causes, self.num_nodes))
        else:
            # 靠近根：选入度小的/拓扑靠前的
            y_candidates = list(range(self.num_causes, min(self.num_causes + self.num_outputs * 4, self.num_nodes)))

        # 采样 y 索引
        if len(y_candidates) < self.num_outputs:
            y_indices = random.sample(range(self.num_causes, self.num_nodes), self.num_outputs)
        else:
            y_indices = random.sample(y_candidates, self.num_outputs)

        # X 候选：排除 y 与根因（很多场景希望 X 是“中间/可观测特征”而非 exogenous；如需包含根因，可放开 num_causes 的限制）
        X_pool = [i for i in range(self.num_causes, self.num_nodes) if i not in y_indices]

        if self.in_clique and len(X_pool) >= self.num_features:
            # 采用“相邻块”（近似 clique / 同层邻近）
            # 做法：在拓扑序上找一个连续窗口
            topo = [n for n in self.topo_order if (n in X_pool)]
            if len(topo) >= self.num_features:
                start = random.randint(0, len(topo) - self.num_features)
                X_indices = topo[start : start + self.num_features]
            else:
                X_indices = random.sample(X_pool, self.num_features)
        else:
            # 随机采样
            if len(X_pool) < self.num_features:
                # 不够就允许从根因里补齐
                extra_pool = [i for i in range(0, self.num_nodes) if i not in y_indices]
                X_indices = random.sample(extra_pool, self.num_features)
            else:
                X_indices = random.sample(X_pool, self.num_features)

        if self.sort_features:
            X_indices = sorted(X_indices)

        X = all_nodes[:, X_indices]
        y = all_nodes[:, y_indices]
        return X, y



# class MLPSCM(nn.Module):
#     """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).
    
#     Copied from : https://github.com/soda-inria/tabicl/blob/main/src/tabicl/prior/mlp_scm.

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
