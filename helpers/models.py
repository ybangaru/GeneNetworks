"""
Provides variations of GNN models and loss functions used for node classification and graph classification.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from helpers import logger


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, no_edge_types, aggr="add"):
        super(GINConv, self).__init__()

        # Multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding = torch.nn.Embedding(no_edge_types, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, no_edge_types):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = no_edge_types - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        if x_j.shape[1] != edge_attr.shape[1]:
            if self.aggr == "add":
                logger.error("Aggregation type is add, but the dimensions of x_j and edge_attr do not match.")
                return
        if not self.aggr == "add":
            # TODO: Implement other aggregation types when node, edge embbings have different dimensions
            raise NotImplementedError("Aggregation type other than add is not implemented yet.")
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, no_edge_types, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(no_edge_types, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr, no_edge_types):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = no_edge_types - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, no_edge_types=None, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding = torch.nn.Embedding(no_edge_types, heads * emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, no_edge_types):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = no_edge_types - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.emb_dim)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim)
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, no_edge_types, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(no_edge_types, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, no_edge_types):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = no_edge_types - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        num_feat (int): number of features besides node type
        emb_dim (int): dimensionality of embeddings
        node_embedding_output (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(
        self,
        num_layer=3,
        num_node_type=None,
        num_feat=38,
        emb_dim=256,
        emb_dim_edge=256,
        node_embedding_output="last",
        drop_ratio=0,
        gnn_type="gin",
        edge_type_dict=None,
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.node_embedding_output = node_embedding_output
        self.no_edge_types = len(edge_type_dict)

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = torch.nn.Embedding(num_node_type, emb_dim)
        self.feat_embedding = torch.nn.Linear(num_feat, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.feat_embedding.weight.data)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim_edge, no_edge_types=len(edge_type_dict), aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(
                    GCNConv(
                        emb_dim_edge,
                        no_edge_types=len(edge_type_dict),
                    )
                )
            elif gnn_type == "gat":
                self.gnns.append(
                    GATConv(
                        emb_dim_edge,
                        no_edge_types=len(edge_type_dict),
                    )
                )
            elif gnn_type == "graphsage":
                self.gnns.append(
                    GraphSAGEConv(
                        emb_dim_edge,
                        no_edge_types=len(edge_type_dict),
                    )
                )

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim_edge))

    def forward(self, *argv):
        if len(argv) == 3:
            node_feat_mask = None
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            # Support for GNNExplainer
            x, edge_index, edge_attr, node_feat_mask = (
                argv[0],
                argv[1],
                argv[2],
                argv[3],
            )
        elif len(argv) == 1:
            data = argv[0]
            node_feat_mask = None
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # Note that the first feature should always be cell type
        x = self.x_embedding(x[:, 0].long()) + self.feat_embedding(x[:, 1:].float())
        if node_feat_mask is not None:
            x = x * node_feat_mask

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, self.no_edge_types)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.node_embedding_output == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.node_embedding_output == "last":
            node_representation = h_list[-1]
        elif self.node_embedding_output == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.node_embedding_output == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_pred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        num_node_type (int): number of unique node/cell types
        num_feat (int): number of features besides node type
        emb_dim (int): dimensionality of embeddings
        num_additional_feat (int): number of additional subgraph-level features
        num_node_tasks (int): number of node-level tasks (for only center cells)
        num_graph_tasks (int): number of graph-level tasks
        node_embedding_output (str): one of "last", "concat", "max" or "sum".
        drop_ratio (float): dropout rate
        graph_pooling (str): one of "sum", "mean", "max", "attention", "set2set"
        gnn_type: one of "gin", "gcn", "graphsage", "gat"

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(
        self,
        num_layer=2,
        num_node_type=None,
        num_feat=38,
        emb_dim=256,
        emb_dim_edge=256,
        num_additional_feat=0,
        num_node_tasks=15,
        num_graph_tasks=2,
        node_embedding_output="last",
        drop_ratio=0,
        graph_pooling="mean",
        gnn_type="gin",
        return_node_embedding=False,
        edge_types=None,
    ):
        super(GNN_pred, self).__init__()
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.emb_dim_edge = emb_dim_edge
        self.num_node_tasks = num_node_tasks
        self.num_graph_tasks = num_graph_tasks
        self.return_node_embedding = return_node_embedding

        self.gnn = GNN(
            num_layer,
            num_node_type,
            num_feat,
            emb_dim,
            emb_dim_edge,
            node_embedding_output=node_embedding_output,
            drop_ratio=drop_ratio,
            gnn_type=gnn_type,
            edge_type_dict=edge_types,
        )

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if node_embedding_output == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if node_embedding_output == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For node and graph predictions
        self.mult = 1
        if graph_pooling[:-1] == "set2set":
            self.mult *= 2
        if node_embedding_output == "concat":
            self.mult *= self.num_layer + 1

        node_embedding_dim = self.mult * self.emb_dim
        if self.num_graph_tasks > 0:
            self.graph_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, self.num_graph_tasks),
            )

        if self.num_node_tasks > 0:
            self.node_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, self.num_node_tasks),
            )

    def from_pretrained(self, model_file, strict_bool):
        self.gnn.load_state_dict(torch.load(model_file), strict=strict_bool)

    def forward(self, data, node_feat_mask=None, return_node_embedding=False):
        gnn_args = [data.x, data.edge_index, data.edge_attr]
        if node_feat_mask is not None:
            assert node_feat_mask.shape[0] == data.x.shape[0]
            gnn_args.append(node_feat_mask.to(data.x.device))
        batch = data.batch if "batch" in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        node_representation = self.gnn(*gnn_args)
        if "additional_feat" in data:
            additional_feat = data.additional_feat[batch]
            node_representation = torch.cat([node_representation, additional_feat], 1)

        return_vals = []
        if self.num_node_tasks > 0:
            if "center_node_index" not in data:
                node_pred = self.node_pred_module(node_representation)
            else:
                center_node_index = (
                    [data.center_node_index] if isinstance(data.center_node_index, int) else data.center_node_index
                )
                center_node_rep = node_representation[center_node_index]
                node_pred = self.node_pred_module(center_node_rep)
            return_vals.append(node_pred)
        if self.num_graph_tasks > 0:
            graph_pred = self.graph_pred_module(self.pool(node_representation, batch))
            return_vals.append(graph_pred)
        if self.return_node_embedding:
            return_vals.append(node_representation)
        return return_vals


# Loss functions
class BinaryCrossEntropy(torch.nn.Module):
    """Weighted binary cross entropy loss function"""

    def __init__(self, **kwargs):
        super(BinaryCrossEntropy, self).__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_pred, y, w):
        return (self.loss_fn(y_pred, y) * w).mean()


class CoxSGDLossFn(torch.nn.Module):
    """Cox SGD loss function"""

    def __init__(self, top_n=2, regularizer_weight=0.05, **kwargs):
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight
        super(CoxSGDLossFn, self).__init__(**kwargs)

    def forward(self, y_pred, length, event):
        assert y_pred.shape[0] == length.shape[0] == event.shape[0]
        n_samples = y_pred.shape[0]
        pair_mat = (length.reshape((1, -1)) - length.reshape((-1, 1)) > 0) * event.reshape((-1, 1))

        if self.top_n > 0:
            p_with_rand = pair_mat * (1 + torch.rand_like(pair_mat))
            rand_thr_ind = torch.argsort(p_with_rand, axis=1)[:, -(self.top_n + 1)]
            rand_thr = p_with_rand[(torch.arange(n_samples), rand_thr_ind)].reshape((-1, 1))
            pair_mat = pair_mat * (p_with_rand > rand_thr)

        valid_sample_is = torch.nonzero(pair_mat.sum(1)).flatten()
        pair_mat[(valid_sample_is, valid_sample_is)] = 1

        score_diff = y_pred.reshape((1, -1)) - y_pred.reshape((-1, 1))
        score_diff_row_max = torch.max(score_diff, axis=1, keepdims=True).values
        loss = (torch.exp(score_diff - score_diff_row_max) * pair_mat).sum(1)
        loss = (score_diff_row_max[:, 0][valid_sample_is] + torch.log(loss[valid_sample_is])).sum()

        regularizer = torch.abs(pair_mat.sum(0) * y_pred.flatten()).sum()
        loss += self.regularizer_weight * regularizer
        return loss


# Baseline methods - sklearn style
class MLP_pred(object):
    def __init__(
        self,
        n_feat=None,
        n_layers=3,
        n_hidden=256,
        n_tasks=None,
        gpu=False,
        task="classification",  # classification or cox
        balanced=True,  # used for classification loss
        top_n=0,  # used for cox regression loss
        regularizer_weight=0.005,
        **kwargs
    ):
        self.n_feat = n_feat
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.gpu = gpu
        self.task = task
        self.balanced = balanced
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight

    def build_model(self):
        assert self.n_feat is not None
        assert self.n_tasks is not None
        modules = [torch.nn.Linear(self.n_feat, self.n_hidden)]
        for _ in range(self.n_layers - 1):
            modules.append(torch.nn.LeakyReLU())
            modules.append(torch.nn.Linear(self.n_hidden, self.n_hidden))
        modules.append(torch.nn.LeakyReLU())
        modules.append(torch.nn.Linear(self.n_hidden, self.n_tasks))
        self.model = torch.nn.Sequential(*modules)

        if self.gpu:
            self.model = self.model.cuda()

    def fit(self, X, y, w=None, batch_size=0, n_epochs=200, lr=0.001):
        assert len(X.shape) == 2
        self.n_feat = X.shape[1]
        if len(y.shape) == 1:
            self.n_tasks = 1
            y = y.reshape((-1, 1))
            w = w.reshape((-1, 1)) if w is not None else w
        else:
            self.n_tasks = y.shape[1]

        batch_size = batch_size if batch_size > 0 else X.shape[0]

        dataset = [torch.from_numpy(X).float(), torch.from_numpy(y).float()]
        if w is not None:
            dataset.append(torch.from_numpy(w).float())
        dataset = TensorDataset(*dataset)
        loader = DataLoader(dataset, batch_size=batch_size)

        self.build_model()
        self.model.zero_grad()
        self.model.train()

        if self.task == "classification":
            pos_weight = (1 - y.mean(0)) / y.mean(0) if self.balanced else np.ones((y.shape[1],))
            self.loss_fn = torch.nn.BCEWithLogitsLoss(torch.from_numpy(pos_weight).float())
        elif self.task == "cox":
            self.loss_fn = CoxSGDLossFn(top_n=self.top_n, regularizer_weight=self.regularizer_weight)
        else:
            raise ValueError("Task type unknown")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(n_epochs):
            for batch in loader:
                if self.gpu:
                    batch = [item.cuda() for item in batch]
                y_pred = self.model(batch[0])
                if self.task == "classification":
                    loss = self.loss_fn(y_pred, batch[1])
                elif self.task == "cox":
                    loss = self.loss_fn(y_pred, batch[1], batch[2])

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

    def predict_proba(self, X, batch_size=0):
        assert len(X.shape) == 2
        assert X.shape[1] == self.n_feat

        batch_size = batch_size if batch_size > 0 else X.shape[0]
        dataset = TensorDataset(torch.from_numpy(X).float())
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()

        y_pred = []
        for _X in loader:
            if isinstance(_X, list):
                _X = _X[0]
            if self.gpu:
                _X = _X.cuda()
            y_pred.append(self.model(_X).cpu().data.numpy())

        y_pred = np.concatenate(y_pred)
        y_pred = 1 / (1 + np.exp(-y_pred))
        return np.squeeze(np.stack([1 - y_pred, y_pred], -1))

    def predict(self, X, **kwargs):
        y_pred = self.predict_proba(X, **kwargs)
        return np.argmax(y_pred, -1)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        sd = self.model.state_dict()
        torch.save(sd, path)


def load_partial_state_dict(model, state_dict):
    """Helper function for loading pre-trained weight"""
    own_state = model.state_dict()
    loaded_layers = set()
    for name, param in state_dict.items():
        if name not in own_state:
            print("skipping weight for %s" % name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
        loaded_layers.add(name)
    print("missing: %s" % str(own_state.keys() - loaded_layers))
    return
