import numpy as np
import torch
import torch.nn as nn


class AttnRNN(nn.Module):
    '''AttnRNN implemented with same initializations'''
    def __init__(self, input_size, hidden_size, dropout=0.0, max_seq_len=50, K=1, mode="attnlstm"):
        super(AttnRNN, self).__init__()
        """
        :param hidden_size: a scalar for outputs vector size
        :param K: a scalar for previous size
        """
        self._hidden_size = hidden_size
        self.K = K
        self.is_tree = True
        self.mode = mode.lower()
        self.name = mode.lower()
        self.max_seq_len = max_seq_len

        self.rnnCell = nn.RNNCell(input_size, hidden_size)
        self.LSTMCell = nn.LSTMCell(input_size, hidden_size)
        self.AttnLSTMCell = AttnLSTMCell(input_size, hidden_size, max_seq_len, K)
        self.drop = nn.Dropout(p=dropout)
        #TODO: init lstm weights and biases
        nn.init.orthogonal_(self.rnnCell.weight_ih)
        nn.init.orthogonal_(self.rnnCell.weight_hh)

        nn.init.zeros_(self.rnnCell.bias_ih)
        nn.init.zeros_(self.rnnCell.bias_hh)
    
    @property
    def state_size(self):
        return (self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, input, states=None):
        """
        :param input: Tensor of shape [batch_size, sequence_length, input_size]
                        `batch_size` will be preserved
                        `input_size` must be static
        :param states: None create zero, else use input
                        states contain cells and hiddens
                        cells and hiddens is a list, K element,
                        element shape is [batch_size, outputs_size]
        :param scope: a String
        :return: Hidden states [batch_size, sequence_length, outputs_size], tuple of last cell and last hidden state
        """
        batch_size = input.size()[0]
        seq_len = input.size()[1]

        outputs_size = self._hidden_size
        outputs = []

        if states == None:
            cells = []
            hiddens = []
            for _ in range(self.K):
                cell = torch.zeros((batch_size, outputs_size))
                hidden = torch.zeros((batch_size, outputs_size))
                if(input.is_cuda):
                    cell = cell.cuda()
                    hidden = hidden.cuda()
                cells.append(cell)
                hiddens.append(hidden)
            states = (cells, hiddens)

        cells, hiddens = states
        for idx in range(seq_len):
            if(self.mode == 'rnn'):
                output = self.rnnCell(input[:,idx,:], hiddens[-1])
                cell = output
            elif(self.mode == 'lstm'):
                output, cell = self.LSTMCell(input[:,idx,:], (hiddens[-1],cells[-1]))
            elif(self.mode == 'attnlstm'):
                output, cell = self.AttnLSTMCell(input[:,idx,:], states)
            cells.append(cell)
            hiddens.append(output)
            cells = cells[1:self.K + 1]
            hiddens = hiddens[1:self.K + 1]
            states = (cells, hiddens)
            outputs.append(output)
            
        outputs = torch.stack(outputs,dim=1)
        outputs = self.drop(outputs)
        return outputs, states

class AttnLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len, attn_size):
        super(AttnLSTMCell, self).__init__()
        """
        :param hidden_size: a scalar for outputs vector size
        :param K: a scalar for previous size
        """
        
        self.f_x_lin = nn.Linear(input_size,hidden_size)
        self.o_x_lin = nn.Linear(input_size,hidden_size)
        self.i_x_lin = nn.Linear(input_size,hidden_size)
        self.u_x_lin = nn.Linear(input_size,hidden_size)
        
        self.f_h_lin = nn.Linear(hidden_size,hidden_size)
        self.o_h_lin = nn.Linear(hidden_size,hidden_size)
        self.i_h_lin = nn.Linear(hidden_size,hidden_size)
        self.u_h_lin = nn.Linear(hidden_size,hidden_size)
        
        nn.init.orthogonal_(self.f_x_lin.weight)
        nn.init.orthogonal_(self.o_x_lin.weight)
        nn.init.orthogonal_(self.i_x_lin.weight)
        nn.init.orthogonal_(self.u_x_lin.weight)
        
        nn.init.zeros_(self.f_x_lin.bias)
        nn.init.zeros_(self.o_x_lin.bias)
        nn.init.zeros_(self.i_x_lin.bias)
        nn.init.zeros_(self.u_x_lin.bias)
        
        nn.init.orthogonal_(self.f_h_lin.weight)
        nn.init.orthogonal_(self.o_h_lin.weight)
        nn.init.orthogonal_(self.i_h_lin.weight)
        nn.init.orthogonal_(self.u_h_lin.weight)
       
        nn.init.zeros_(self.f_h_lin.bias)
        nn.init.zeros_(self.o_h_lin.bias)
        nn.init.zeros_(self.i_h_lin.bias)
        nn.init.zeros_(self.u_h_lin.bias)
        
        self.h_k_lins = nn.ModuleList()
        for i in range(max_seq_len):
            current = nn.Linear(hidden_size,hidden_size)
            nn.init.eye_(current.weight)
            nn.init.zeros_(current.bias)
            self.h_k_lins.append(current)
        
        # Attention weights
        self.attnW =   nn.Parameter(torch.normal(mean=torch.zeros((hidden_size, attn_size)),std=0.1))
        self.attnb =   nn.Parameter(torch.normal(mean=torch.zeros((1, attn_size)),std=0.1))
        self.attnW_u = nn.Parameter(torch.normal(mean=torch.zeros((attn_size, 1)),std=0.1))
        

    def forward(self, input, states):
        
        (hiddens, cells) = states
        
        f = self.f_x_lin(input) + self.f_h_lin(hiddens[-1])
        o = self.o_x_lin(input) + self.o_h_lin(hiddens[-1])
        i = self.i_x_lin(input) + self.i_h_lin(hiddens[-1])
        i_gt = torch.sigmoid(i)

        hs = list()
        for k, (cell, hid) in enumerate(zip(cells, hiddens)):
            h_k = self.h_k_lins[k](hid*i_gt)
            hs.append(h_k)
        
        # Attention mechanism
        if isinstance(hs, tuple):
            hs = torch.cat(hs, 2)
        elif isinstance(hs, list):
            hs = torch.stack(hs,1)
        
        batch_size = hs.size()[0]
        seq_len = hs.size()[1]
        hid_size = hs.size()[2]
        
        inp_rshp = hs.reshape(batch_size*seq_len, hid_size)
        u = torch.tanh(torch.mm(inp_rshp, self.attnW) + self.attnb)
        uv = torch.mm(u, self.attnW_u)
        exps = torch.exp(uv).reshape(batch_size, seq_len)
        alphas = exps / torch.sum(exps, 1).reshape(batch_size, 1)

        # Output of RNN is reduced with attention vector
        u_h = torch.sum(hs * alphas.reshape(-1, seq_len, 1), 1)
        
        u = self.u_x_lin(input) + u_h
        cell_next = cells[-1] * torch.sigmoid(f) + torch.tanh(u) * (1 - torch.sigmoid(f))
        hidden_next = torch.tanh(cell_next) * torch.sigmoid(o)
        outputs = hidden_next, cell_next
            
        return outputs