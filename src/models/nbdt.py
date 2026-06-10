import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDecisionTree(nn.Module):
    """
    Vectorized Soft Decision Tree classifier layer.
    Replaces the final linear layer of a neural network to create a Neural-Backed Decision Tree.
    """
    def __init__(self, in_features, num_classes=8, depth=3):
        super(SoftDecisionTree, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.depth = depth
        
        self.num_leaves = 2 ** depth
        self.num_internal_nodes = self.num_leaves - 1
        
        # Routing nodes parameters: w^T x + b
        # We can implement this as a single Linear layer for vectorization
        self.routing_fc = nn.Linear(in_features, self.num_internal_nodes)
        
        # Leaf distributions: probability of each class at each leaf
        # Initialized randomly and learned via backpropagation
        self.leaf_distributions = nn.Parameter(torch.randn(self.num_leaves, num_classes))
        
        # Build path matrix: maps leaf paths to routing decisions
        # Row index: leaf index (0 to num_leaves-1)
        # Column index: internal node index (0 to num_internal_nodes-1)
        # Path matrix contains +1 (must go left), -1 (must go right), 0 (node not in path)
        self.register_buffer('path_matrix', self._build_path_matrix())
        
    def _build_path_matrix(self):
        """Constructs the path routing matrix for soft decision tree nodes."""
        path_matrix = torch.zeros(self.num_leaves, self.num_internal_nodes)
        for leaf_idx in range(self.num_leaves):
            current_node = 0
            # Leaf index represented in binary defines the path
            for d in range(self.depth):
                # Check if we go left (0) or right (1) at current depth
                go_right = (leaf_idx >> (self.depth - 1 - d)) & 1
                if go_right:
                    # Going right: path needs (1 - p) -> we store -1
                    path_matrix[leaf_idx, current_node] = -1
                    current_node = 2 * current_node + 2 # Right child index
                else:
                    # Going left: path needs p -> we store +1
                    path_matrix[leaf_idx, current_node] = 1
                    current_node = 2 * current_node + 1 # Left child index
        return path_matrix
        
    def forward(self, x):
        # x shape: (Batch, in_features)
        batch_size = x.shape[0]
        
        # Compute routing probabilities for all internal nodes
        # d_i = sigmoid(w_i^T x + b_i)
        # Shape: (Batch, num_internal_nodes)
        decisions = torch.sigmoid(self.routing_fc(x))
        
        # Expand decisions to shape: (Batch, num_leaves, num_internal_nodes)
        decisions_expanded = decisions.unsqueeze(1).expand(-1, self.num_leaves, -1)
        
        # Compute path probabilities for all leaves
        # For a leaf, path probability is product of:
        #   p_i if path_matrix is +1
        #   (1 - p_i) if path_matrix is -1
        #   1 if path_matrix is 0
        
        # We can compute this using a masking trick:
        # positive path mask: 1 where path_matrix is 1, else 0
        # negative path mask: 1 where path_matrix is -1, else 0
        pos_mask = (self.path_matrix == 1).float().unsqueeze(0)
        neg_mask = (self.path_matrix == -1).float().unsqueeze(0)
        
        # Compute probabilities per node
        node_probs = (decisions_expanded * pos_mask) + ((1 - decisions_expanded) * neg_mask) + (1 - pos_mask - neg_mask)
        
        # Product along internal nodes axis to get leaf path probabilities
        # Shape: (Batch, num_leaves)
        leaf_probs = torch.prod(node_probs, dim=2)
        
        # Normalize leaf class distributions (Softmax along classes)
        # Shape: (num_leaves, num_classes)
        leaf_class_dist = F.softmax(self.leaf_distributions, dim=1)
        
        # Final class probability is weighted sum of leaf class distributions
        # Shape: (Batch, num_classes)
        out = torch.matmul(leaf_probs, leaf_class_dist)
        
        # Use log_softmax or raw probabilities?
        # Standard CrossEntropyLoss in PyTorch expects raw logits, so we take log of probabilities
        # Adding a small epsilon to avoid log(0)
        logits = torch.log(out + 1e-8)
        return logits

class NBDT(nn.Module):
    """
    Neural-Backed Decision Tree wrapper.
    Paper: "NBDT: Neural-Backed Decision Trees" (ICLR 2021)
    Uses LeNet-5 features backplane followed by a Soft Decision Tree routing layer.
    """
    def __init__(self, in_channels=4, num_classes=8, image_size=(64, 64)):
        super(NBDT, self).__init__()
        
        # Feature extractor based on LeNet-5 conv layers
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate FC dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            out = self.pool1(F.relu(self.conv1(dummy)))
            out = self.pool2(F.relu(self.conv2(out)))
            self.fc_input_dim = out.view(1, -1).size(1)
            
        # Dense representation layer
        self.fc1 = nn.Linear(self.fc_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        
        # Decision Tree depth depends on the number of classes
        # For 8 classes, depth = 3. For 2 classes, depth = 1.
        depth = 3 if num_classes == 8 else 1
        
        # Replaces standard linear classifier with Soft Decision Tree
        self.nbdt_classifier = SoftDecisionTree(in_features=84, num_classes=num_classes, depth=depth)
        
    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = out.view(out.size(0), -1)
        
        out = F.relu(self.fc1(out))
        features = F.relu(self.fc2(out))
        
        # Soft decision tree classification
        logits = self.nbdt_classifier(features)
        return logits
