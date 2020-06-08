import numpy as np
import pandas as pd
import torch
#from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from torch.autograd import Variable
from torch.autograd import Variable, grad
from torch.functional import F
from torch_geometric.data import Data, DataLoader, NeighborSampler
from torch_geometric.nn import MetaLayer, MessagePassing
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
#from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange
from torch import sparse
import torch_sparse as ts
#!/usr/bin/env python
# coding: utf-8

def get_cluster():
   import socket
   host = socket.gethostname().lower()
   if host.startswith('adroit'):
     return 'adroit'
   elif host.startswith('tiger'):
     return 'tiger'
   else:
     return ''

def load_graph_data(realization=0, cutoff=30):
    try:
        cur_data = pd.read_hdf('halos_%d.h5'%(realization,))
    except:
        from generate_halo_data_nv import generate_data
        generate_data(realization, get_cluster())
        cur_data = pd.read_hdf('halos_%d.h5'%(realization,))

# # Now, let's connect nearby halos:

    xyz = np.array([cur_data.x, cur_data.y, cur_data.z]).T
    tree = KDTree(xyz)

# ## Let's see what a good radius is. Let's aim for ~8 particles or so for average

    region_of_influence = cutoff

    #plt.hist(tree.query_radius(xyz, region_of_influence, count_only=True)-1, bins=31);
    #plt.xlabel('Number with')
    #plt.ylabel('Number of neighbors')

# ## So, let's create the adjacency matrix:

    neighbors = tree.query_radius(xyz, region_of_influence, sort_results=True, return_distance=True)[0]

    all_edges = []
    for j in range(len(neighbors)):
        if len(neighbors[j]) == 1:
            continue
        #Receiving is second!
        cur = np.array([
            neighbors[j][1:],
            np.ones(len(neighbors[j])-1)*j],
            dtype=np.int64)
        
        all_edges.append(cur)
        
    all_edges = np.concatenate(all_edges, axis=1)

# # Now let's put this data into PyTorch:

    X_raw = torch.from_numpy(np.array(cur_data['x y z vx vy vz M14'.split(' ')]))
    y_raw = torch.from_numpy(np.array(cur_data[['delta']]))
    pos_scale = X_raw[:, :3].std(0).mean(0)
    pos_mean = 500
    vel_scale = X_raw[:, 3:6].std(0).mean(0)
    M14_scale = X_raw[:, 6].std()
    X = X_raw.clone()
    X[:,  :3] = (X[:,  :3] - pos_mean)/pos_scale
    X[:, 3:6] = (X[:, 3:6])/vel_scale
    X[:, 6] = (X[:, 6])/M14_scale
    edge_index = torch.LongTensor(torch.from_numpy(all_edges))

    cur_data['z'].min()

# Which nodes are far enough from the edge?

    nodes_far_from_edge = np.product([
            (
                (region_of_influence        < cur_data[dim]) &
                (1000 - region_of_influence > cur_data[dim])
            )
            for dim in 'x y z'.split(' ')
        ],
        0).astype(np.float32)

# We'll include this in the y-vector as a simple multiplier against the loss for bad nodes:

    y = torch.cat([
        y_raw,
        torch.from_numpy(nodes_far_from_edge)[:, None]
    ], dim=1)
    graph_data = Data(
        X,
        edge_index=edge_index,
        y=y)


    return {
        'graph': graph_data,
        'column_description': 'x columns are [x, y, z, vx, vy, vz, M]; everything has been scaled to be std=1. y columns are [bias, mask], where mask=1 indicates that the node should be used as a receiver for training; mask=0 indicates that the node is too close to the edge. Multiply the node-wise loss by the mask during training.',
        'pos_scale': pos_scale,
        'vel_scale': vel_scale,
        'M14_scale': M14_scale}


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        quick_mlp = lambda n_in, n_out: Seq(
            Lin(n_in, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, n_out)
        )
        
        self.msg_fnc = quick_mlp(2*(n_f-3) + 3, msg_dim)
        self.node_fnc = quick_mlp(msg_dim+n_f-3, ndim)
        
        self.n = 32
        
    def forward(self, g):
        #x is [n, n_f]
        x = g.x
        edge_index = g.edge_index
#         return self.propagate(edge_index, x=x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        dx = x_i[:, :3] - x_j[:, :3]
        r2 = (dx**2).sum(1) + 1
        tmp = torch.cat([
            dx,
            x_i[:, 3:], x_j[:, 3:]], dim=1) # tmp has shape [E, 2 * in_channels]
        return self.msg_fnc(tmp) #* (100/r2[:, None])
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x[:self.n, 3:], aggr_out[:self.n]], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]

def create_graph_network(hidden=300, msg_dim=100, output_dim=1):
    # We'll use the L1 regularization:

    aggr = 'add'
    test = '_l1_'
    dim = output_dim #Predict a scalar value
    n_f = 3*2+1#X.shape[1]

# # We use a custom data loader for the graphs for fast training:
# # Let's declare the model:

    ogn = GN(n_f, msg_dim, dim, hidden=hidden, aggr=aggr)

    return ogn

def new_loss(self, g, cur_len, augment=True, regularization=1e-2):
    
    self.n = cur_len
    mask = g.y[:cur_len, [1]]
    base_loss = torch.sum(torch.abs(g.y[:cur_len, [0]] - self(g))*mask)*cur_len/(mask.sum()+1e-8)
    
    s1 = g.x[g.edge_index[0]]
    s2 = g.x[g.edge_index[1]]
    m12 = self.message(s1, s2)
    #Want one loss value per row of g.y:
    normalized_l05 = torch.sum(torch.abs(m12))*cur_len*32/len(m12)
    
    return base_loss, regularization * normalized_l05

def do_training(
        ogn, graph,
        lr=1e-3, total_epochs=100,
        batch_per_epoch=1500, weight_decay=1e-8,
        batch=32, l1=1e-2):

    idx = graph.edge_index.cuda()
    X = graph.x.cuda()
    y = graph.y.cuda()
    N = graph.x.shape[0]
    device = torch.device('cuda')
    v = torch.ones(idx.shape[1], device=device)
    mat = sparse.IntTensor(idx, v, torch.Size([N, N]))
    mat2 = ts.tensor.SparseTensor.from_torch_sparse_coo_tensor(mat, has_value=False)
    row, col, _ = mat2.csr()
    
    # Set up optimizer:
    init_lr = lr
    opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=weight_decay)

    sched = OneCycleLR(opt, max_lr=init_lr,
                       steps_per_epoch=batch_per_epoch,#len(trainloader),
                       epochs=total_epochs, final_div_factor=1e5)

    all_losses = []
    epoch = 0

    for epoch in trange(epoch, total_epochs):
        ogn.cuda()
        total_loss = 0.0
        i = 0
        num_items = 0

        while i < batch_per_epoch:
            opt.zero_grad()

            node_idx = torch.randint(0, N-1, (batch,), device=device)
            neighbor_idx = torch.cat([col[row[node_idx[i]]:row[node_idx[i]+1]] for i in range(batch)])

            new_node_idx = torch.cat([
                torch.ones(
                    row[node_idx[i]+1] - row[node_idx[i]], dtype=int, device=device
                )*i for i in range(batch)])
            new_neighbor_idx = torch.arange(batch, batch+len(neighbor_idx), device=device, dtype=int)

            Xcur = torch.cat([X[node_idx], X[neighbor_idx]], dim=0)
            ycur = torch.cat([y[node_idx], y[neighbor_idx]], dim=0)

            edge_index = torch.cat([new_neighbor_idx[None], new_node_idx[None]])#new_node_idx[None], new_neighbor_idx[None]])
            
            g = Data(
                 x=Xcur,
                 y=ycur,
                 edge_index=edge_index
            )
            
            loss, reg = new_loss(ogn, g, batch, regularization=l1)
            ((loss + reg)/int(batch+1)).backward()
            
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += batch

        cur_loss = total_loss/num_items
        all_losses.append(cur_loss)
        print(cur_loss, flush=True)

    return all_losses

def get_messages(ogn, trainloader, n_msg=250):
    print("Warning: this function assumes that only a single message component dominates", flush=True)
    all_msg_input = []
    all_msgs = []
    all_msg_sums = []
    all_nodes = []
    all_outputs = []

    X = trainloader.data.x
    y = trainloader.data.y
    batch = trainloader.batch_size

    i = 0

    for subgraph in trainloader():
        
        n_offset = len(subgraph.n_id)
        cur_len = n_offset
        cur_edge_index = subgraph.blocks[0].edge_index.clone()
        cur_edge_index[0] += n_offset
        g = Data(
            x=torch.cat((
                X[subgraph.n_id],
                X[subgraph.blocks[0].n_id])).cuda(),
            y=torch.cat((
                y[subgraph.n_id],
                y[subgraph.blocks[0].n_id])).cuda(),
            edge_index=cur_edge_index.cuda()
        )
        
        s1 = g.x[g.edge_index[0]]
        s2 = g.x[g.edge_index[1]]
        msg_input = torch.cat([s1[:, :3] - s2[:, :3], s1[:, 3:], s2[:, 3:]], dim=1)
        
        raw_msg = ogn.msg_fnc(msg_input)
        msg_input = msg_input.detach().cpu().numpy()
        all_msg_input.append(msg_input)
        
        best_msg_idx = np.argmax(raw_msg.std(0).detach().cpu().numpy())
        best_msgs = raw_msg[:, best_msg_idx].detach().cpu().numpy()
        all_msgs.append(best_msgs)
        
        associated_sum_message = np.array([
            raw_msg[np.argwhere(g.edge_index[1].detach().cpu().numpy() == i).T].sum(0)[best_msg_idx].detach().cpu().numpy()
            for i in range(batch)
        ])
        all_msg_sums.append(associated_sum_message)
        node = g.x[list(range(batch))]
        output = ogn(g)
        all_nodes.append(node.detach().cpu().numpy())
        all_outputs.append(output.detach().cpu().numpy())
        
        i += 1
        if i > n_msg:
            break
        
    all_msg_input = np.concatenate(all_msg_input)
    all_msgs = np.concatenate(all_msgs)
    all_msg_sums = np.concatenate(all_msg_sums)
    all_nodes = np.concatenate(all_nodes)
    all_outputs = np.concatenate(all_outputs)

    #plt.scatter(
    #    x=np.arange(raw_msg.std(0).shape[0]),
    #    y=np.log10(np.sort(raw_msg.std(0).detach().cpu().numpy())),
    #    s=3
    #)

    msg_func_data = pd.DataFrame({**{
        'dx dy dz vx1 vy1 vz1 M1 vx2 vy2 vz2 M2'.split(' ')[i]: all_msg_input[:, i] for i in range(all_msg_input.shape[1])
    },
                                  **{
        'message': all_msgs
    }})

    node_func_data = pd.DataFrame({**{
        'x y z vx vy vz M'.split(' ')[i]: all_nodes[:, i] for i in range(7)
    },
                  **{
        'message': all_msg_sums, 'output': all_outputs[:, 0]
    }})


    idx_node = np.arange(node_func_data.shape[0])
    np.random.shuffle(idx_node)

    idx_msg = np.arange(msg_func_data.shape[0])
    np.random.shuffle(idx_msg)
    
    return {
        'node_function': node_func_data.iloc[idx_node],#.iloc[:5000].to_csv('node_func.csv');
        'msg_function': msg_func_data.iloc[idx_msg]#.iloc[:5000].to_csv('msg_func.csv')
        }


