import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

class AttnRNN(nn.Module):
    '''AttnRNN implemented with same initializations'''
    def __init__(self, input_size, num_units, max_seq_len, K=1, mode="lstm"):
        super(AttnRNN, self).__init__()
        """
        :param num_units: a scalar for outputs vector size
        :param K: a scalar for previous size
        """
        self._num_units = num_units
        self.K = K
        self.is_tree = True
        self.mode = mode
        self.name = mode
        self.max_seq_len = max_seq_len
        print("create a "+mode)

        self.rnnCell = nn.RNNCell(input_size, num_units)
        self.LSTMCell = nn.LSTMCell(input_size, num_units)
        self.AttnLSTMCell = AttnLSTMCell(input_size, num_units, max_seq_len, K)
        
        #TODO: init lstm weights and biases
        nn.init.orthogonal_(self.rnnCell.weight_ih)
        nn.init.orthogonal_(self.rnnCell.weight_hh)

        nn.init.zeros_(self.rnnCell.bias_ih)
        nn.init.zeros_(self.rnnCell.bias_hh)
    
    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def forward(self, input, states=None):

        if(states is None):
            hidden = torch.zeros((input.size()[0],self._num_units)) # batch-size x hidden-size
            cell = torch.zeros((input.size()[0],self._num_units))
            states = ([hidden],[cell])
        hidden_next, cell_next = self.AttnLSTMCell(input,states)
        
        return hidden_next, cell_next

    def dynamic_rnn(self, inputs, states=None, scope=None):
        """
        :param inputs: Tensor of shape [batch_size, times, input_size]
                        `batch_size` will be preserved (known)
                        `times` sequence length
                        `input_size` must be static (known)
        :param states: None create zero, else use input
                        states contain cells and hiddens
                        cells and hiddens is a list, K element,
                        element shape is [batch_size, outputs_size]
        :param scope: a String
        :return: Tensor of shape [batch_size, times, outputs_size]
        """
        batch_size = inputs.get_shape().as_list()[0]
        times = inputs.get_shape().as_list()[1]
        input_size = inputs.get_shape().as_list()[2]
        outputs_size = self._num_units
        outputs = []
        with tf.variable_scope(scope or type(self).__name__):
            if states == None:
                cells = []
                hiddens = []
                for i in range(self.K):
                    cell = tf.zeros([batch_size, outputs_size])
                    hidden = tf.zeros([batch_size, outputs_size])
                    cells.append(cell)
                    hiddens.append(hidden)
                states = (cells, hiddens)
            else:
                cells, hiddens = states
            inps = tf.unstack(inputs,times,1)
            for idx, inp in enumerate(inps):
                if idx > 0: tf.get_variable_scope().reuse_variables()
                output, cell = self.__call__(inp, states)
                cells.append(cell)
                hiddens.append(output)
                cells = cells[1:self.K + 1]
                hiddens = hiddens[1:self.K + 1]
                states = (cells, hiddens)
                outputs.append(output)
        return outputs, states


    def dynamic_rnn_v2(self, inputs, states=None, scope=None):
        """
        :param inputs:Tensor List. shape is times*[batch_size, input_size]
                    `batch_size` will be preserved (known)
                    `times` sequence length
                    `input_size` must be static (known)
        :param states: None create zero, else use input
                        states contain cells and hiddens
                        cells and hiddens is a list, child_size element,
                        element shape is [batch_size, outputs_size]
        :param scope: a String
        :return: outputs: Tensor List, shape is times*[batch_size, outputs_size].
        """
        batch_size = inputs[0].get_shape().as_list()[0]
        input_size = inputs[0].get_shape().as_list()[1]
        outputs_size = self._num_units
        outputs = []
        with tf.variable_scope(scope or type(self).__name__):
            if states == None:
                cells = []
                hiddens = []
                for i in range(self.K):
                    cell = tf.zeros([batch_size, outputs_size])
                    hidden = tf.zeros([batch_size, outputs_size])
                    cells.append(cell)
                    hiddens.append(hidden)
                states = (cells, hiddens)
            else:
                cells, hiddens = states
            for idx, inp in enumerate(inputs):
                if idx > 0: tf.get_variable_scope().reuse_variables()
                output, cell = self.__call__(inp, states)
                cells.append(cell)
                hiddens.append(output)
                cells = cells[1:self.K + 1]
                hiddens = hiddens[1:self.K + 1]
                states = (cells, hiddens)
                outputs.append(output)
        return outputs, states

class AttnLSTMCell(nn.Module):
    def __init__(self, input_size, num_units, max_seq_len, attn_size):
        super(AttnLSTMCell, self).__init__()
        """
        :param num_units: a scalar for outputs vector size
        :param K: a scalar for previous size
        """
        
        self.f_x_lin = nn.Linear(input_size,num_units)
        self.o_x_lin = nn.Linear(input_size,num_units)
        self.i_x_lin = nn.Linear(input_size,num_units)
        self.u_x_lin = nn.Linear(input_size,num_units)
        
        self.f_h_lin = nn.Linear(num_units,num_units)
        self.o_h_lin = nn.Linear(num_units,num_units)
        self.i_h_lin = nn.Linear(num_units,num_units)
        self.u_h_lin = nn.Linear(num_units,num_units)
        
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
        
        self.h_k_lins = list()
        for i in range(max_seq_len):
            current = nn.Linear(num_units,num_units)
            nn.init.eye_(current.weight)
            nn.init.zeros_(current.bias)
            self.h_k_lins.append(current)
        
        # Attention weights
        self.attnW =   nn.Parameter(torch.normal(mean=torch.zeros((num_units, attn_size)),std=0.1))
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