
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as nn
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub, MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys


class GCNConv(nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # print(f'x shape: {x.shape}')

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Multiply with weights
        x = self.lin(x)

        # Calculate normalization
        # print(edge_index)
        # print(edge_index.shape)

        sources, targets = edge_index
        deg = degree(sources, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[sources] * deg_inv_sqrt[targets]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train(model, optimizer, data, n_epochs, plot=False):
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, _ = get_acc_and_loss(data, model)
        val_acc, test_loss = get_acc_and_loss(data, model, type='val')

        train_accuracies.append(train_acc)
        train_losses.append(loss.item())
        val_accuracies.append(val_acc)
        val_losses.append(test_loss)

        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.
              format(epoch, loss.item(), train_acc, val_acc))

    if plot:
        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Validation losses")
        plt.xlabel("# Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.show()

        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(val_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()


def get_acc_and_loss(data, model, type='train'):
    model.eval()

    correct = 0
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]).item()
    pred = out.max(dim=1)[1]

    if type == 'train':
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        acc = correct / (len(data.y[data.train_mask]))
    elif type == 'val':
        correct += pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
        acc = correct / (len(data.y[data.val_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / (len(data.y[data.test_mask]))

    return acc, loss


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


if __name__ == '__main__':
    # dataset = TUDataset(root='/tmp/Enzymes', name='Enzymes')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')  # Cora, CiteSeer, or PubMed

    print(dataset.data)
    print_info_about_dataset(dataset)
    # plot_dataset(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset.data.to(device)

    model = Net(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(model, optimizer, data, n_epochs=100, plot=True)

    test_acc, _ = get_acc_and_loss(data, model, type='test')
    print(f'---- Accuracy on test set: {test_acc}')
