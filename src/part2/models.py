import torch
from torch import nn
from torch.nn import functional as F


def _prepare_graph(x, edge_index, batch):
    if x.ndim != 2:
        raise ValueError("x must have shape [num_nodes, num_features]")
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError("batch must have one graph index per node")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    graph_ids, batch = torch.unique(batch, sorted=True, return_inverse=True)
    num_graphs = graph_ids.numel()
    counts = torch.bincount(batch, minlength=num_graphs)
    starts = torch.cumsum(counts, dim=0) - counts
    roots = starts
    return edge_index.long(), batch.long(), roots.long(), num_graphs


def _mean_pool(x, batch, num_graphs):
    pooled = x.new_zeros((num_graphs, x.shape[-1]))
    pooled.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).to(x.dtype).clamp_min_(1)
    return pooled / counts.unsqueeze(-1)


def _max_pool(x, batch, num_graphs):
    pooled = x.new_full((num_graphs, x.shape[-1]), -torch.inf)
    pooled.scatter_reduce_(0, batch[:, None].expand_as(x), x, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def _add_self_loops(edge_index, num_nodes):
    nodes = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat((edge_index, torch.stack((nodes, nodes))), dim=1)


class _GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        edges = _add_self_loops(edge_index, num_nodes)
        source, target = edges
        degree = x.new_zeros(num_nodes)
        degree.index_add_(0, target, x.new_ones(target.shape[0]))
        norm = degree[source].clamp_min_(1).rsqrt() * degree[target].clamp_min_(1).rsqrt()
        transformed = self.linear(x)
        output = transformed.new_zeros(transformed.shape)
        output.index_add_(0, target, transformed[source] * norm.unsqueeze(-1))
        return output + self.bias


class _SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        source, target = edge_index
        neighbors = x.new_zeros(x.shape)
        degree = x.new_zeros(num_nodes)
        if source.numel():
            neighbors.index_add_(0, target, x[source])
            degree.index_add_(0, target, x.new_ones(target.shape[0]))
        neighbors = neighbors / degree.clamp_min_(1).unsqueeze(-1)
        return self.self_linear(x) + self.neighbor_linear(neighbors)


class _GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, out_features))
        self.attn_target = nn.Parameter(torch.empty(heads, out_features))
        width = heads * out_features if concat else out_features
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        edges = _add_self_loops(edge_index, num_nodes)
        source, target = edges
        h = self.linear(x).view(num_nodes, self.heads, self.out_features)
        scores = (h[source] * self.attn_source).sum(-1) + (h[target] * self.attn_target).sum(-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        max_scores = scores.new_full((num_nodes, self.heads), -torch.inf)
        max_scores.scatter_reduce_(0, target[:, None].expand_as(scores), scores, reduce="amax", include_self=True)
        weights = (scores - max_scores[target]).exp()
        denominator = scores.new_zeros((num_nodes, self.heads))
        denominator.index_add_(0, target, weights)
        weights = weights / denominator[target].clamp_min_(1e-12)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = h.new_zeros((num_nodes, self.heads, self.out_features))
        output.index_add_(0, target, h[source] * weights.unsqueeze(-1))
        if self.concat:
            output = output.reshape(num_nodes, self.heads * self.out_features)
        else:
            output = output.mean(dim=1)
        return output + self.bias


class RootOnlyMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch=None, root_index=None):
        _, batch, roots, _ = _prepare_graph(x, edge_index, batch)
        if root_index is not None:
            roots = root_index.to(device=x.device, dtype=torch.long).reshape(-1)
        return self.classifier(x[roots])


class _GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout):
        super().__init__()
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def classify_nodes(self, x, batch, roots, num_graphs):
        pooled = torch.cat((_mean_pool(x, batch, num_graphs), x[roots]), dim=-1)
        return self.classifier(pooled)


class GCN(_GraphClassifier):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, num_classes, dropout)
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        widths = [in_channels] + [hidden_channels] * num_layers
        self.layers = nn.ModuleList(_GCNLayer(widths[i], widths[i + 1]) for i in range(num_layers))

    def forward(self, x, edge_index, batch=None, root_index=None):
        edge_index, batch, roots, num_graphs = _prepare_graph(x, edge_index, batch)
        if root_index is not None:
            roots = root_index.to(device=x.device, dtype=torch.long).reshape(-1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x, edge_index))
        return self.classify_nodes(x, batch, roots, num_graphs)


class GraphSAGE(_GraphClassifier):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, num_classes, dropout)
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        widths = [in_channels] + [hidden_channels] * num_layers
        self.layers = nn.ModuleList(_SAGELayer(widths[i], widths[i + 1]) for i in range(num_layers))

    def forward(self, x, edge_index, batch=None, root_index=None):
        edge_index, batch, roots, num_graphs = _prepare_graph(x, edge_index, batch)
        if root_index is not None:
            roots = root_index.to(device=x.device, dtype=torch.long).reshape(-1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x, edge_index))
        return self.classify_nodes(x, batch, roots, num_graphs)


class GAT(_GraphClassifier):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2, num_layers=2, heads=4, dropout=0.5):
        super().__init__(in_channels, hidden_channels, num_classes, dropout)
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        layers = []
        width = in_channels
        for _ in range(num_layers - 1):
            layers.append(_GATLayer(width, hidden_channels // heads, heads, True, dropout))
            width = hidden_channels
        layers.append(_GATLayer(width, hidden_channels, heads, False, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch=None, root_index=None):
        edge_index, batch, roots, num_graphs = _prepare_graph(x, edge_index, batch)
        if root_index is not None:
            roots = root_index.to(device=x.device, dtype=torch.long).reshape(-1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, edge_index))
        return self.classify_nodes(x, batch, roots, num_graphs)


class BiGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.top_down = nn.ModuleList((_GCNLayer(in_channels, hidden_channels), _GCNLayer(hidden_channels, hidden_channels)))
        self.bottom_up = nn.ModuleList((_GCNLayer(in_channels, hidden_channels), _GCNLayer(hidden_channels, hidden_channels)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch=None, root_index=None):
        edge_index, batch, roots, num_graphs = _prepare_graph(x, edge_index, batch)
        if root_index is not None:
            roots = root_index.to(device=x.device, dtype=torch.long).reshape(-1)
        top = F.dropout(x, p=self.dropout, training=self.training)
        bottom = top
        reverse_edges = edge_index.flip(0)
        for layer in self.top_down:
            top = F.relu(layer(top, edge_index))
            top = F.dropout(top, p=self.dropout, training=self.training)
        for layer in self.bottom_up:
            bottom = F.relu(layer(bottom, reverse_edges))
            bottom = F.dropout(bottom, p=self.dropout, training=self.training)
        representation = torch.cat((top[roots], bottom[roots], _max_pool(top, batch, num_graphs), _max_pool(bottom, batch, num_graphs)), dim=-1)
        return self.classifier(representation)


def build_model(name, in_channels, hidden_channels=128, num_classes=2, dropout=0.5, **kwargs):
    key = name.lower().replace("_", "").replace("-", "")
    models = {
        "mlp": RootOnlyMLP,
        "rootonlymlp": RootOnlyMLP,
        "gcn": GCN,
        "sage": GraphSAGE,
        "graphsage": GraphSAGE,
        "gat": GAT,
        "bigcn": BiGCN,
    }
    if key not in models:
        raise ValueError(f"Unknown model: {name}")
    model_class = models[key]
    if model_class in (RootOnlyMLP, BiGCN):
        kwargs.pop("num_layers", None)
        kwargs.pop("heads", None)
    elif model_class is not GAT:
        kwargs.pop("heads", None)
    return model_class(in_channels, hidden_channels=hidden_channels, num_classes=num_classes, dropout=dropout, **kwargs)
