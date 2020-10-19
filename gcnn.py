import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time


class GrahpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GrahpConv, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x, A):
        agg = torch.mm(A, x)
        return self.W(agg)


class GCNN(nn.Module):
    def __init__(self, in_dim, hid_dim=32, out_dim=10):
        super(GCNN, self).__init__()
        self.gconv1 = GrahpConv(in_channels=in_dim, out_channels=hid_dim)
        self.gconv2 = GrahpConv(hid_dim, out_dim)

    def forward(self, x, A):
        h = F.relu(self.gconv1(x, A))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gconv2(h, A)
        return h


def train(model, optimizer, data, A, n_epochs, plot=False, device=None):
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    x = data.x.to(device)
    targets = data.y.to(device)

    train_mask = data.train_mask.to(device)
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(x, A)
        loss = F.cross_entropy(out[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        train_acc, _ = get_acc_and_loss(x, targets, model, A, device=device)
        val_acc, val_loss = get_acc_and_loss(x, targets, model, A, type='val', device=device)

        train_accuracies.append(train_acc)
        train_losses.append(loss.item())
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch}, Loss: {loss.item():.3f}, Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}', )

    print(f'Training done in {(time.time() - start):.1f}s')

    if plot:
        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Validation losses")
        plt.xlabel("# Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.savefig('./outputs/loss_plot.png')
        plt.show()

        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(val_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.savefig('./outputs/accuracy_plot.png')
        plt.show()


def create_adjacency_matrix(num_nodes, edge_index, add_self_loops=True, normalize=True, device=None):
    """Creates adjacency matrix from pytorch_geometric edge_index"""
    adj = torch.zeros((num_nodes, num_nodes))
    edges = torch.stack((edge_index[0], edge_index[1]), 1)
    for (source_i, target_i) in edges:
        adj[source_i, target_i] = 1

    if add_self_loops:
        adj = adj + torch.eye(adj.size(0))

    if normalize:
        d = adj.sum(1)
        adj = adj / d.view(-1, 1)

    return adj.to(device)


def get_acc_and_loss(x, targets, model, A, type='train', device=None):
    model.eval()

    correct = 0
    out = model(x, A)
    pred = out.max(dim=1)[1]

    if type == 'train':
        mask = data.train_mask.to(device)
    elif type == 'val':
        mask = data.val_mask.to(device)
    else:
        mask = data.test_mask.to(device)

    loss = F.cross_entropy(out[mask], targets[mask]).item()
    correct += pred[mask].eq(targets[mask]).sum().item()
    acc = correct / (len(targets[mask]))

    return acc, loss


def print_info_about_dataset(dataset):
    """ Expects dataset from torch_geometric.datasets"""
    try:
        data = dataset.data
        print('------ PRINTING INFO ------')
        print('>> INFO')
        print(f'\tLength of dataset (number of graphs): {len(dataset)}')
        if len(dataset) == 1:
            print(f'\tData is one big graph.')
        else:
            print(f'\tData contains multiple graphs.')

        print(f'\tNum nodes: {data.num_nodes}')
        print(f'\tNum node features: {dataset.num_node_features}')
        print(f'\tNum classes (node or graph classes): {dataset.num_classes}')
        print(f'\tEdges contains attributes: {"False" if data.edge_attr is None else "True"}')
        print(f'\tTarget (y) shape: {data.y.shape}')

        if len(dataset) == 1:
            print(f'\tNode feature matrix shape: {data.x.shape}')
            print(f'\tContains isolated nodes: {data.contains_isolated_nodes()}')
            print(f'\tContains self-loops: {data.contains_self_loops()}')
            print(f'\tEdge_index shape: {data.edge_index.shape}')
            print(f'\tEdges are directed: {data.is_directed()}')
        else:
            print(f'\tPrinting info about first graph:')
            print(f'\t{dataset[0]}')
        print('>> END')

    except:
        print('Some prints failed.')


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
        'node_size': 30,
        'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = Planetoid(root='/tmp/Cora', name='Cora')  # Cora, CiteSeer, or PubMed
    print(dataset.data)
    print_info_about_dataset(dataset)

    data = dataset.data
    num_nodes = data.num_nodes
    num_features = dataset.num_node_features

    x = data.x.to(device)  # Node features
    y = data.y.to(device)  # Node classes
    num_targets = len(torch.unique(y))
    print(f'Num classes: {num_targets}')

    A = create_adjacency_matrix(num_nodes, data.edge_index, device=device)
    model = GCNN(num_features, hid_dim=16, out_dim=num_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    if torch.cuda.is_available():
        model.cuda()

    train(model, optimizer, data, A, n_epochs=100, plot=True, device=device)

    test_acc, _ = get_acc_and_loss(x, y, model, A, type='test', device=device)
    print(f'---- Accuracy on test set: {test_acc}')
