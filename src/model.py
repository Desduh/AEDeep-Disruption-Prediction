# model.py
import torch
import torch.nn as nn

class PeakPredictorLSTM(nn.Module):
    def __init__(self, model_config):
        """
        Initializes the LSTM model for peak prediction.
        
        Args:
            model_config (dict): A dictionary with model parameters:
                                 input_dim, projection_dim, hidden_dim,
                                 num_layers, dropout.
        """
        super().__init__()
        self.projection_dim = model_config['projection_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Input Projection Layer
        self.input_projection = nn.Linear(model_config['input_dim'], self.projection_dim)
        
        # Intermediate SiLU activation
        self.silu = nn.SiLU()

        # 2. LSTM Layers
        # We now need to access each LSTM layer individually to pass the hidden state.
        self.lstm_layers = nn.ModuleList()
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(self.projection_dim, self.hidden_dim, batch_first=True))
        # Subsequent LSTM layers
        for _ in range(self.num_layers - 1):
            # Input to subsequent layers is hidden_dim, not projection_dim
            self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))

        # 3. Dropout Layer
        self.dropout = nn.Dropout(model_config['dropout'])

        # 4. Output Projection Layer
        # It takes the final hidden state as input
        self.output_projection = nn.Linear(self.hidden_dim, 1)
        
        # 5. Final Sigmoid Activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for TRAINING. Processes an entire sequence at once.
        This remains unchanged to not affect the training script.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Apply input projection and activation
        x = self.input_projection(x)
        x = self.silu(x)

        # Pass through LSTM layers with intermediate SiLU and Dropout
        # For training, we don't need to manage hidden states manually
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
                x = self.silu(x)
        
        last_time_step_output = x[:, -1, :]
        output = self.output_projection(last_time_step_output)
        output = self.sigmoid(output)

        return output

    def init_hidden(self):
        """
        Initializes the hidden and cell states for all LSTM layers.
        To be called at the beginning of an inference sequence.
        """
        hidden_states = []
        for _ in range(self.num_layers):
            # (h, c) for each layer: shape (1, batch_size=1, hidden_dim)
            h = torch.zeros(1, 1, self.hidden_dim).to(self.device)
            c = torch.zeros(1, 1, self.hidden_dim).to(self.device)
            hidden_states.append((h, c))
        return hidden_states

    def step(self, x_t, prev_hidden_states):
        """
        Forward pass for a SINGLE time step (INFERENCE).
        
        Args:
            x_t (torch.Tensor): Input for the current time step. 
                                Shape: (batch_size=1, seq_len=1, input_dim)
            prev_hidden_states (list of tuples): List of (h, c) tuples from the previous step.
            
        Returns:
            tuple: (prediction, new_hidden_states)
        """
        # 1. Apply input projection and activation
        x_t = self.input_projection(x_t)
        x_t = self.silu(x_t)

        # 2. Pass through LSTM layers one by one, managing state
        new_hidden_states = []
        current_input = x_t
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Pass input and the specific hidden state for this layer
            output, (h_new, c_new) = lstm_layer(current_input, prev_hidden_states[i])
            
            # The output of this layer becomes the input for the next (after dropout/activation)
            current_input = output
            if i < self.num_layers - 1:
                current_input = self.dropout(current_input)
                current_input = self.silu(current_input)
                
            new_hidden_states.append((h_new, c_new))

        # 3. The final output is from the last layer
        final_output = current_input
        
        # 4. Apply output projection and activation
        # Squeeze to remove the seq_len dimension of 1
        prediction = self.output_projection(final_output.squeeze(1))
        prediction = self.sigmoid(prediction)

        return prediction, new_hidden_states