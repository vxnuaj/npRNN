# 2 layer stacked lstm with a linear layer + softmax output. `see test.py`` for more

import numpy as np
from ops import OPS as ops

class LSTM:

    '''
    Notes:
    
    we assume input size of n x d where n is batch size and d is input embedding size.
    '''
    
    def __init__(
        self,
        n_neurons: tuple,
        in_dim: int,
        seq_len: int,
        batch_size = 1,
        linear_out = True
        ):
  
   
        self.n_lstm_layers = 2
        self.n_neurons = n_neurons
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.linear_out = linear_out
       
        if self.linear_out:
            i = 1
        else:
            i = 0 
        
        assert len(n_neurons) == self.n_lstm_layers + i
       
        internals = self._init_internals() 
       
        self.w_i = internals[0] 
        self.w_f = internals[1]
        self.w_o = internals[2]
        self.w_c = internals[3]
        self.w_hi = internals[4]
        self.w_hf = internals[5]
        self.w_ho = internals[6]
        self.w_hc = internals[7]
        self.w_lin = internals[8]
        self.b_i = internals[9]
        self.b_f = internals[10]
        self.b_o = internals[11]
        self.b_c = internals[12]
        self.b_lin = internals[13]
        self.I = internals[14]
        self.F = internals[15]
        self.O = internals[16]
        self.C = internals[17]
        self.H = internals[18]
        self.I_node = internals[19]
        self.A = internals[20]
        
    def _init_internals(self):
        
        # init input, forget, and output gate weights.
        w_i = [0 for _ in range(self.n_lstm_layers)]
        w_f = [0 for _ in range(self.n_lstm_layers)]
        w_o = [0 for _ in range(self.n_lstm_layers)]
        w_c = [0 for _ in range(self.n_lstm_layers)]
        
        b_i = [0 for _ in range (self.n_lstm_layers)]  
        b_f = [0 for _ in range (self.n_lstm_layers)]  
        b_o = [0 for _ in range (self.n_lstm_layers)]  
        b_c = [0 for _ in range (self.n_lstm_layers)]  
        
        # init input, forget, and outpug gate hidden weights
        w_hi = [0 for _ in range(self.n_lstm_layers)]
        w_hf = [0 for _ in range(self.n_lstm_layers)]
        w_ho = [0 for _ in range(self.n_lstm_layers)]
        w_hc = [0 for _ in range(self.n_lstm_layers)]
        
                
     
        for layer in range(self.n_lstm_layers):
            if layer == 0:
                w_i[layer] = np.random.randn(
                    self.in_dim, 
                    self.n_neurons[layer]
                    )
        
                w_f[layer] = np.random.randn(
                    self.in_dim, 
                    self.n_neurons[layer]
                    )               
                
                w_o[layer] = np.random.randn(
                    self.in_dim, 
                    self.n_neurons[layer]
                    )               
                
                w_c[layer] = np.random.randn(
                    self.in_dim, 
                    self.n_neurons[layer]
                    )                              
                
                w_hi[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )               
                 
                w_hf[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )               
                  
                w_ho[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )   

                w_hc[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )                              
                
                b_i[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )          
                
                b_f[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )           
                
                b_o[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )                               

                b_c[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )
                
            else:
                 
                w_i[layer] = np.random.randn(
                    self.n_neurons[layer - 1],
                    self.n_neurons[layer]
                    )
        
                w_f[layer] = np.random.randn(
                    self.n_neurons[layer - 1], 
                    self.n_neurons[layer]
                    )               
                
                w_o[layer] = np.random.randn(
                    self.n_neurons[layer - 1], 
                    self.n_neurons[layer]
                    )               
               
                w_c[layer] = np.random.randn(
                    self.n_neurons[layer - 1],
                    self.n_neurons[layer]
                ) 
                
                w_hi[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )               
                 
                w_hf[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )               
                  
                w_ho[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )   

                w_hc[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                    )   
 
                b_i[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )          
                
                b_f[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )           
                
                b_o[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )                               
               
                b_c[layer] = np.zeros(
                    shape = (1, self.n_neurons[layer])
                )                  
                
        I = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons
        ] 
        
        I_node = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons
        ] 
        
        F = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons
        ] 
         
        O = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons
        ]
        
        C = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons
        ]
        
        H = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)]
            for n_neuron in self.n_neurons 
        ]
            
        w_lin = np.random.randn(self.n_neurons[-2], self.n_neurons[-1])
        b_lin = np.zeros(shape = (1, self.n_neurons[-1]))
       
        A = [0 for _ in range(self.seq_len)]
            
        return w_i, w_f, w_o, w_c, w_hi, w_hf, w_ho, w_hc, w_lin, b_i, b_f, b_o, b_c, b_lin, I, F, O, C, H, I_node, A
        
    
    def forward(self, X):
      
        for t in range(self.seq_len):
            for layer in range(self.n_lstm_layers + 1):
       
                if layer == 0:

                    if t != 0:
                        hidden_state = self.H[layer][t - 1]
                        cell_state = self.C[layer][t-1]
                    else:
                        hidden_state = None
                        cell_state = None

                    # initial gate computation

                    self.F[layer][t] = self._forget_gate(
                        w_in = self.w_f[layer],
                        w_hidden = self.w_hf[layer],
                        bias = self.b_f[layer],
                        hidden_state = hidden_state,
                        X = X[:, :, t],
                    )
                   
                    self.I[layer][t] = self._input_gate(
                        w_in = self.w_i[layer],
                        w_hidden = self.w_hi[layer],
                        bias = self.b_i[layer],
                        hidden_state = hidden_state,
                        X = X[:, :, t] 
                    ) 
                   
                    self.I_node[layer][t] = self._input_node(
                        w_in = self.w_c[layer],
                        w_hidden = self.w_hc[layer],
                        bias = self.b_c[layer],
                        hidden_state=hidden_state,
                        X = X[:, :, t]
                    )
                    
                    self.O[layer][t] = self._output_gate(
                        w_in = self.w_o[layer],
                        w_hidden = self.w_ho[layer],
                        bias = self.b_o[layer],
                        hidden_state=hidden_state,
                        X = X[:, :, t]
                    )
              
                    # cell state and output computation 
               
                    self.C[layer][t] = self._cell_state(
                        F = self.F[layer][t],
                        I = self.I[layer][t],
                        I_node = self.I_node[layer][t],
                        cell_state = cell_state
                    )

                    self.H[layer][t] = self._hidden_state(
                        cell_state = self.C[layer][t],
                        O = self.O[layer][t]
                    )
                    
                elif layer == 1:

                    if t != 0:
                        hidden_state = self.H[layer][t-1]
                        cell_state = self.C[layer][t-1]
                    else:
                        hidden_state = None
                        cell_state = None

                    self.F[layer][t] = self._forget_gate(
                        w_in = self.w_f[layer],
                        w_hidden = self.w_hf[layer],
                        bias = self.b_f[layer],
                        X = self.H[layer - 1][t],
                        hidden_state = hidden_state
                    )
                   
                    self.I[layer][t] = self._input_gate(
                        w_in = self.w_i[layer],
                        w_hidden = self.w_hi[layer],
                        bias = self.b_i[layer],
                        X = self.H[layer - 1][t],
                        hidden_state = hidden_state
                    )
                  
                    
                    self.I_node[layer][t] = self._input_node(
                        w_in = self.w_c[layer],
                        w_hidden = self.w_hc[layer],
                        bias = self.b_c[layer],
                        X = self.H[layer - 1][t],
                        hidden_state = hidden_state
                    )
                    
                    self.O[layer][t] = self._output_gate(
                        w_in = self.w_o[layer],
                        w_hidden = self.w_ho[layer],
                        bias = self.b_o[layer],
                        X = self.H[layer - 1][t],
                        hidden_state = hidden_state
                    )
              
                    # cell state and output computation 
               
                    self.C[layer][t] = self._cell_state(
                        F = self.F[layer][t],
                        I = self.I[layer][t],
                        I_node = self.I_node[layer][t],
                        cell_state = cell_state
                    )

                    self.H[layer][t] = self._hidden_state(
                        cell_state = self.C[layer][t],
                        O = self.O[layer][t]
                    )
         
                if layer == 2:
                    
                    self.A[t] = self._linear_layer(self.H[layer - 1][t], self.w_lin, self.b_lin) 
          
        return self.H[-1][-1] # return final output shape at final time step.
                     
    def _forget_gate(self, w_in, w_hidden, bias, X, hidden_state = None):
   
        z = np.dot(X, w_in) + bias
        
        if hidden_state is not None:
            z += np.dot(hidden_state, w_hidden) 
            
        return ops.sigmoid(z) 
    
    def _input_gate(self, w_in, w_hidden, bias, X, hidden_state = None):
        
        z = np.dot(X, w_in) + bias
        
        if hidden_state is not None:
            z += np.dot(hidden_state, w_hidden) 
            
        return ops.sigmoid(z)   
    
    def _input_node(self, w_in, w_hidden, bias, X, hidden_state = None):
        
        z = np.dot(X, w_in) + bias
       
        if hidden_state is not None:
 
            z += np.dot(hidden_state, w_hidden) 
            
        return ops.tanh(z)
    
    def _output_gate(self, w_in, w_hidden, bias, X, hidden_state = None):
        
        z = np.dot(X, w_in) + bias
        
        if hidden_state is not None:
            z += np.dot(hidden_state, w_hidden) 
            
        return ops.sigmoid(z)
   
    def _cell_state(self, F, I, I_node, cell_state = None):

        if cell_state is not None: 
            assert F.shape == cell_state.shape, f"Forget gate and cell state are not same dims. Got {F.shape} and {cell_state.shape} respectively"
            cell_state = F * cell_state
        else:
            cell_state = 0  
            
        cell_state += I * I_node
        
        return cell_state
        
    def _hidden_state(self, cell_state, O):
        
        c_2 = ops.tanh(cell_state) 
        out = c_2 * O 
            
        return out
    
    def _linear_layer(self, X, W, bias):
       
        z = np.dot(X, W) + bias
        
        return ops.softmax(z)

    
   
    