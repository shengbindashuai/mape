import random
from typing import Any, Dict, Tuple, List, Set, Iterable

import networkx as nx
import torch
import torch.nn as nn

from .utils import GaussianNoise, XSampler

class MLPSCM(nn.Module):
    """
    用 NetworkX 构建的结构因果模型（SCM）来生成表格数据。
    （包含改进的 _select_X_y：优先选择 bottleneck y 与 K-hop Markov blanket 的 X）
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
        edge_prob: float = 0.5,        # 边存在概率（越大图越密）
        # 生成函数与噪声
        init_std: float = 1.0,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        device: str = "cpu",
        # Markov blanket hops (新增)
        mb_hops: int = 1,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.init_std = init_std
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.device = device
        self.mb_hops = mb_hops

        # 图规模：保证足够多的中间变量可以抽取 X 和 y
        if num_nodes is None:
            num_nodes = max(num_causes + 2 * num_features + num_outputs, num_causes + 8)
        assert num_nodes >= num_causes + num_features + num_outputs, \
            "num_nodes 至少需要覆盖根因与 X/y 的数量。"
        self.num_nodes = num_nodes
        self.edge_prob = random.uniform(0.1, 0.5)

        # 输入采样器（只用于根因）
        # 注意：这里假定 XSampler 在别处定义并导入，可与原实现保持一致
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

        # 选择 X 与 y 的索引（使用改进后的选择逻辑）
        X, y = self._select_X_y(all_nodes)

        # NaN 保护
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    # -------------------------
    # 改进后的 X/y 抽取策略
    # -------------------------
    def _select_X_y(self, all_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        all_nodes: (seq_len, num_nodes)
        新逻辑：
        1) Y 选择：优先挑选既有较大入度又有较大出度的“bottleneck”节点（用 indeg*outdeg 得分），
           若不足则回退到原有 sinks/靠前节点策略，确保能选到 num_outputs。
        2) X 选择：优先从 y 的 Markov blanket（parents U children U parents(children)）开始，
           再做 K-hop 扩展（self.mb_hops），优先采样这些节点；若不足再从其它候选补齐。
        """
        # ----- 参数与实用函数 -----
        mb_hops = getattr(self, "mb_hops", 1)

        def k_hop_neighbors(seed_nodes: Iterable[int], k: int) -> Set[int]:
            """在有向图上做无向的 k-hop 邻居扩展（通过 predecessors 和 successors）。"""
            visited = set(seed_nodes)
            frontier = set(seed_nodes)
            for _ in range(k):
                if not frontier:
                    break
                next_frontier = set()
                for n in frontier:
                    preds = set(self.G.predecessors(n))
                    succs = set(self.G.successors(n))
                    for nb in (preds | succs):
                        if nb not in visited:
                            next_frontier.add(nb)
                            visited.add(nb)
                frontier = next_frontier
            return visited

        def markov_blanket_of(nodes: Iterable[int]) -> Set[int]:
            """标准 Markov blanket：parents U children U parents(children)"""
            mb = set()
            for y in nodes:
                parents = set(self.G.predecessors(y))
                children = set(self.G.successors(y))
                parents_of_children = set()
                for c in children:
                    parents_of_children |= set(self.G.predecessors(c))
                mb |= parents | children | parents_of_children
            return mb

        # ----- 1) 选择 y 索引（优先 bottleneck） -----
        candidate_nodes = [n for n in range(self.num_causes, self.num_nodes)]

        # 计算度分数（避免根因）
        deg_scores: Dict[int, int] = {}
        for n in candidate_nodes:
            indeg = self.G.in_degree(n)
            outdeg = self.G.out_degree(n)
            deg_scores[n] = int(indeg) * int(outdeg)  # product score：既要入度又要出度

        # 高分候选
        nonzero = [n for n, s in deg_scores.items() if s > 0]

        y_indices: List[int] = []
        if len(nonzero) >= self.num_outputs:
            # 按分数排序（降序），从 top pool 随机抽取最终 y
            sorted_by_score = sorted(nonzero, key=lambda n: deg_scores[n], reverse=True)
            top_pool = sorted_by_score[: max(len(sorted_by_score), self.num_outputs * 3)]
            # 若 top_pool 小于需求也会在后面补齐
            pick_count = min(len(top_pool), self.num_outputs)
            # 为了随机性，从 top_pool 中 sample
            y_indices = random.sample(top_pool, pick_count)
        else:
            # 回退到原始策略
            if getattr(self, "y_is_effect", True):
                sinks = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
                y_candidates = sinks if len(sinks) >= self.num_outputs else candidate_nodes
            else:
                y_candidates = list(range(self.num_causes, min(self.num_causes + self.num_outputs * 4, self.num_nodes)))

            if len(y_candidates) < self.num_outputs:
                y_indices = random.sample(candidate_nodes, self.num_outputs)
            else:
                y_indices = random.sample(y_candidates, self.num_outputs)

        # 确保整数列表且无重复，且补齐至 num_outputs
        y_indices = list(dict.fromkeys(map(int, y_indices)))
        if len(y_indices) < self.num_outputs:
            remaining = [n for n in candidate_nodes if n not in y_indices]
            need = self.num_outputs - len(y_indices)
            if remaining and need > 0:
                y_indices += random.sample(remaining, min(len(remaining), need))

        # ----- 2) 构造 X 候选，优先来自 y 的 K-stage Markov blanket -----
        mb = markov_blanket_of(y_indices)
        extended_mb = set(k_hop_neighbors(mb | set(y_indices), mb_hops))

        # 过滤掉根因与 y 自身（期望 X 是中间/可观测特征）
        preferred_pool = [n for n in extended_mb if (n not in range(0, self.num_causes)) and (n not in y_indices)]

        # 原来的 X_pool（排除 y 与根因）
        X_pool = [i for i in range(self.num_causes, self.num_nodes) if i not in y_indices]

        X_indices: List[int] = []

        # 优先从 preferred_pool 填充
        if len(preferred_pool) >= self.num_features:
            if self.in_clique:
                # 拓扑上取连续窗口（若可行）
                topo_pref = [n for n in self.topo_order if n in preferred_pool]
                if len(topo_pref) >= self.num_features:
                    start = random.randint(0, len(topo_pref) - self.num_features)
                    X_indices = topo_pref[start : start + self.num_features]
                else:
                    X_indices = random.sample(preferred_pool, self.num_features)
            else:
                X_indices = random.sample(preferred_pool, self.num_features)
        else:
            # 先用完 preferred_pool，然后从 X_pool 补齐
            X_indices = list(preferred_pool)
            need = self.num_features - len(X_indices)
            remaining = [n for n in X_pool if n not in X_indices]
            if len(remaining) >= need:
                X_indices += random.sample(remaining, need)
            else:
                # 仍不够的话允许从根因或其它节点补齐
                extra_pool = [i for i in range(0, self.num_nodes) if i not in y_indices and i not in X_indices]
                take = min(len(extra_pool), need)
                if take > 0:
                    X_indices += random.sample(extra_pool, take)

        # 最终排序（如需）
        if self.sort_features:
            X_indices = sorted(X_indices)

        X = all_nodes[:, X_indices]
        y = all_nodes[:, y_indices]
        return X, y