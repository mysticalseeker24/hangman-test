import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size=761, hidden_size=256, output_size=26):
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, state, actions_used):
        # Ensure proper reshaping of state
        if state.dim() > 2:
            # Handle 3D state (batch, height, width)
            state_flat = state.view(state.size(0), -1)
        else:
            # Handle 2D state (single observation)
            state_flat = state.view(1, -1)
        
        # Ensure actions are properly shaped
        if actions_used.dim() == 1:
            actions_used = actions_used.unsqueeze(0)
            
        # Calculate total features
        combined_size = state_flat.size(1) + actions_used.size(1)
        
        # Create padding if needed to match expected input size
        if combined_size < self.input_size:
            padding_size = self.input_size - combined_size
            padding = torch.zeros(state_flat.size(0), padding_size, device=state_flat.device)
            x = torch.cat([state_flat, actions_used, padding], dim=1)
        else:
            # If input is larger than expected, truncate it (should be rare)
            x = torch.cat([state_flat, actions_used], dim=1)[:, :self.input_size]
            
        # Forward pass through network
        x = self.relu(self.dropout(self.input_layer(x)))
        x = self.relu(self.dropout(self.fc1(x)))
        return self.fc2(x)
