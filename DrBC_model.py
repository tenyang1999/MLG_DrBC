import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.utils as tgu
import torch.nn as nn
import random

class DrBC():
    
    def __init__(self,EMBEDDING_SIZE,node_default_dim,layer,batch):
        self.LEARNING_RATE = 0.0001
        self.embedding_size = EMBEDDING_SIZE
        self.node_init = node_default_dim
        self.layer_num = layer
        self.batch = batch
        
        self.W0 = nn.init.normal_(torch.Tensor(self.node_init,self.embedding_size),std=0.01).requires_grad_(True)
        self.W1 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        self.W2 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        self.W3 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        self.U1 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        self.U2 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        self.U3 = nn.init.normal_(torch.Tensor(self.embedding_size,self.embedding_size),std=0.01).requires_grad_(True)
        
        self.W4 = nn.init.normal_(torch.Tensor(self.embedding_size,1),std=0.01).requires_grad_(True)
        self.W5 = nn.init.normal_(torch.Tensor(1,1),std=0.01).requires_grad_(True)
        self.activation = nn.ReLU(inplace=False)

    
    def Encoder_Decoder(self,edge_index,num_nodes):
        
        adj, deg  =self.adjcent_with_weight(edge_index,num_nodes)
        
        # initialize [node_cnt,3]
        init_X = torch.ones(num_nodes, 3)
        init_X[:,:1] = deg.reshape(-1,1)
        
        # [node_cnt, embed_dim]
        
        #cur_h = nn.functional.normalize(self.activation(torch.mm(init_X,self.W0)), p=2.0)
        cur_h = self.activation(torch.mm(init_X,self.W0))
        
        sigmoid = nn.Sigmoid()
        
        for L in range(2,self.layer_num):
            
            # neighbor  # [node_cnt, embed_dim]
            Neighbor_h = torch.mm(adj,cur_h)
            
            # GRU cell
            u_t = sigmoid(torch.add(torch.mm(Neighbor_h,self.W1),torch.mm(cur_h,self.U1)))
            r_t = sigmoid(torch.add(torch.mm(Neighbor_h,self.W2),torch.mm(cur_h,self.U2)))
            f_t = sigmoid(torch.add(torch.mm(Neighbor_h,self.W3),torch.mm(torch.mul(cur_h,r_t),self.U3)))
            new_h = torch.add(torch.mul(u_t,f_t),torch.mul((1-u_t),cur_h))
            
            #new_h = nn.functional.normalize(new_h, p=2.0)
            
            # max z_v
            cur_h =  torch.minimum(cur_h, new_h)
            
        
            
        #Decoder
        # two layer MLP
        
        y = torch.mm(self.activation(torch.mm(cur_h,self.W4)),self.W5)
        y = torch.flatten(y)
        return y
    
    def train(self,episode,num_nodes):
        loss_list = []
        for i in range(episode):
            
            #print(f'episode:{i}')
            G = nx.powerlaw_cluster_graph(n = num_nodes, m=10, p=0.1, seed=None)
            for j in range(self.batch):
                sub_G = nx.powerlaw_cluster_graph(n=num_nodes, m=10, p=0.1, seed=None)
                nd = np.array(sub_G.nodes)+num_nodes*j
                eg = np.array(sub_G.edges)+num_nodes*j

                G.add_nodes_from(nd)
                G.add_edges_from(eg)
                
            # Calculate each node’s exact BC value
            bc = nx.betweenness_centrality(G, normalized=False)
            bc = torch.tensor(list((bc.values()))) 
            bc = torch.log(bc+5.37389E-05)
            
            # Calculate edge index
            x,y = zip(*list(G.edges()))
            edge_index = torch.tensor([x,y])
            
            num_nodes = G.number_of_nodes()
            
            # Encoder : Get each node’s embedding 　
            # Decoder : Compute BC ranking score y
            pred = self.Encoder_Decoder(edge_index,num_nodes)
            
            #  compute loss and update weight
            loss_ = self.loss(num_nodes,bc,pred)
            self.optimal(loss_)
            
            loss_list.append(float(loss_))
            
        return(pred,loss_list)
    
    def predict(self,graph,bc):
        loss_list = []
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index
        pred = self.Encoder_Decoder(edge_index,num_nodes)
        loss = self.loss(num_nodes,bc,pred)
        loss_list.append(float(loss.sum()))
        
        return pred,loss_list
        
    def adjcent_with_weight(self,edge_index,num_nodes):
        
        # compute degree of each node
            row_deg = tgu.degree(edge_index[0],num_nodes = num_nodes)
            col_deg = tgu.degree(edge_index[1],num_nodes = num_nodes)
            deg = row_deg + col_deg

        # compute adjacent matrix
            adj1 = tgu.to_dense_adj(edge_index = edge_index)
            adj2 = tgu.to_dense_adj(edge_index = edge_index[[1,0],:])
            adj = adj1+ adj2
            adj = adj[0]
        
        # Compute neighborhood weight
            row = edge_index[0]
            col = edge_index[1]
            deg_inv_sqrt = (deg+1).pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        #[node_cnt, node_cnt]
            for i in range(num_nodes):
                row_ = edge_index[0][i]
                col_ = edge_index[1][i]
                adj[row_][col_] = norm[i]
                adj[col_][row_] = norm[i]

            return adj, deg
           
    def loss(self, num_nodes,bc,pred):
        # 隨機node pair

        S = torch.tensor([random.randint(0,num_nodes-1) for _ in range(num_nodes*5) ])
        E = torch.tensor([random.randint(0,num_nodes-1) for _ in range(num_nodes*5) ])
        
        preds =  torch.index_select(pred, 0, S)-torch.index_select(pred, 0, E)
        labels = torch.index_select(bc, 0, S)-torch.index_select(bc, 0, E)
        
        sigmoid = nn.Sigmoid()
        
        #loss = torch.mul(-labels,torch.log(sigmoid(preds)))-torch.mul((1-labels),torch.log(1-sigmoid(preds)))
        loss = torch.mul(-sigmoid(labels),torch.log(self.activation(preds)))-torch.mul((1-sigmoid(labels)),torch.log(1-self.activation(preds)))
        loss =  loss.sum()

        return loss
    
    def optimal(self,loss):

        loss.backward()
        
        self.W0.grad[self.W0.grad == float('inf')] = 0
        self.W0 = (torch.add(self.W0,-self.LEARNING_RATE*self.W0.grad))
        self.W0.retain_grad()
        self.W1.grad[self.W1.grad == float('inf')] = 0
        self.W1 = (torch.add(self.W1,-self.LEARNING_RATE*self.W1.grad))
        self.W1.retain_grad()
        self.W2.grad[self.W2.grad == float('inf')] = 0
        self.W2 = (torch.add(self.W2,-self.LEARNING_RATE*self.W2.grad))
        self.W2.retain_grad()
        self.W3.grad[self.W3.grad == float('inf')] = 0
        self.W3 = (torch.add(self.W3,-self.LEARNING_RATE*self.W3.grad))
        self.W3.retain_grad()
        self.U1.grad[self.U1.grad == float('inf')] = 0
        self.U1 = (torch.add(self.U1,-self.LEARNING_RATE*self.U1.grad))
        self.U1.retain_grad()
        self.U2.grad[self.U2.grad == float('inf')] = 0
        self.U2 = (torch.add(self.U2,-self.LEARNING_RATE*self.U2.grad))
        self.U2.retain_grad()
        self.U3.grad[self.U3.grad == float('inf')] = 0
        self.U3 = (torch.add(self.U3,-self.LEARNING_RATE*self.U3.grad))
        self.U3.retain_grad()
        self.W4.grad[self.W4.grad == float('inf')] = 0
        self.W4 = (torch.add(self.W4,-self.LEARNING_RATE*self.W4.grad))
        self.W4.retain_grad()
        self.W5.grad[self.W5.grad == float('inf')] = 0
        self.W5 = (torch.add(self.W5,-self.LEARNING_RATE*self.W5.grad))
        self.W5.retain_grad()
        
    def zero_grad(self):
        self.W0.grad.zero_()
        self.W1.grad.zero_()
        self.W2.grad.zero_()
        self.W3.grad.zero_()
        self.W4.grad.zero_()
        self.W5.grad.zero_()
        self.U1.grad.zero_()
        self.U2.grad.zero_()
        self.U3.grad.zero_()
        
            