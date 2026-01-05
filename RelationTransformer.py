import numpy as np
import math

class RelationTransformer:
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize weights
        self.params = {}
        
        # Embedding projection
        self.params['W_embed'] = self._glorot(input_dim, model_dim)
        
        # Positional encoding (fixed)
        self.pe = self._get_positional_encoding(1000, model_dim)
        
        # Encoder Layers
        for l in range(num_layers):
            # Self Attention
            self.params[f'l{l}_W_q'] = self._glorot(model_dim, model_dim)
            self.params[f'l{l}_W_k'] = self._glorot(model_dim, model_dim)
            self.params[f'l{l}_W_v'] = self._glorot(model_dim, model_dim)
            self.params[f'l{l}_W_o'] = self._glorot(model_dim, model_dim)
            self.params[f'l{l}_ln1_g'] = np.ones(model_dim)
            self.params[f'l{l}_ln1_b'] = np.zeros(model_dim)
            
            # Feed Forward
            self.params[f'l{l}_W_ff1'] = self._glorot(model_dim, model_dim * 4)
            self.params[f'l{l}_b_ff1'] = np.zeros(model_dim * 4)
            self.params[f'l{l}_W_ff2'] = self._glorot(model_dim * 4, model_dim)
            self.params[f'l{l}_b_ff2'] = np.zeros(model_dim)
            self.params[f'l{l}_ln2_g'] = np.ones(model_dim)
            self.params[f'l{l}_ln2_b'] = np.zeros(model_dim)
            
        # Output Head (predict coefficients -1, 0, 1 -> 3 classes)
        self.params['W_out'] = self._glorot(model_dim, 3)
        self.params['b_out'] = np.zeros(3)

    def _glorot(self, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def _get_positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def _layer_norm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b

    def forward(self, x):
        # x shape: [seq_len, input_dim]
        seq_len = x.shape[0]
        
        # Embedding
        h = np.dot(x, self.params['W_embed']) # [seq_len, model_dim]
        
        # Add Positional Encoding
        h = h + self.pe[:seq_len, :]
        
        for l in range(self.num_layers):
            # Multi-Head Attention (Simplified to Single Head for numpy brevity, but logic holds)
            # Q, K, V
            Q = np.dot(h, self.params[f'l{l}_W_q'])
            K = np.dot(h, self.params[f'l{l}_W_k'])
            V = np.dot(h, self.params[f'l{l}_W_v'])
            
            # Scaled Dot-Product Attention
            d_k = self.model_dim
            scores = np.dot(Q, K.T) / np.sqrt(d_k)
            attn = self._softmax(scores)
            context = np.dot(attn, V)
            
            # Output projection
            out = np.dot(context, self.params[f'l{l}_W_o'])
            
            # Residual + Norm
            h = self._layer_norm(h + out, self.params[f'l{l}_ln1_g'], self.params[f'l{l}_ln1_b'])
            
            # Feed Forward
            ff = np.maximum(0, np.dot(h, self.params[f'l{l}_W_ff1']) + self.params[f'l{l}_b_ff1']) # ReLU
            ff = np.dot(ff, self.params[f'l{l}_W_ff2']) + self.params[f'l{l}_b_ff2']
            
            # Residual + Norm
            h = self._layer_norm(h + ff, self.params[f'l{l}_ln2_g'], self.params[f'l{l}_ln2_b'])
            
        # Output Head
        logits = np.dot(h, self.params['W_out']) + self.params['b_out'] # [seq_len, 3]
        probs = self._softmax(logits)
        
        return probs

    def divine_coefficients(self, basis_matrix):
        """
        Uses the transformer to predict coefficients for the basis vectors.
        Returns a vector of coefficients [-1, 0, 1].
        """
        # Normalize input
        basis_norm = basis_matrix / (np.max(np.abs(basis_matrix)) + 1e-8)
        
        probs = self.forward(basis_norm)
        
        # Sample from probabilities
        coeffs = []
        for p in probs:
            # Classes: 0 -> -1, 1 -> 0, 2 -> 1
            choice = np.random.choice([-1, 0, 1], p=p)
            coeffs.append(choice)
            
        return np.array(coeffs)

    def train(self, basis_matrix, target_coeffs_list, iterations=100, positive=True):
        """
        Train the transformer to mimic the target coefficients using evolutionary strategy.
        If positive=True, learn to reproduce these patterns (smooth relations).
        If positive=False, learn to avoid these patterns (partial relations).
        """
        if not target_coeffs_list: return
        
        # Normalize input
        basis_norm = basis_matrix / (np.max(np.abs(basis_matrix)) + 1e-8)
        
        def get_score(model_params):
            # Calculate log likelihood of targets
            # This is a bit complex to do exactly without forward pass modification
            # So we'll just check if the model predicts the correct sign for indices
            # Or simpler: just check if the output probabilities favor the target classes
            
            # We need to run forward pass
            # But we can't easily inject params into forward without changing structure
            # So we rely on self.params being set
            
            probs = self.forward(basis_norm) # [seq_len, 3]
            
            total_log_prob = 0
            for target in target_coeffs_list:
                # target is [seq_len] with values -1, 0, 1
                # Map to indices: -1->0, 0->1, 1->2
                indices = target + 1
                
                # Sum log probs
                for i, idx in enumerate(indices):
                    p = probs[i, int(idx)]
                    if positive:
                        # Maximize probability of matching target (smooth relations)
                        total_log_prob += np.log(p + 1e-10)
                    else:
                        # Minimize probability of matching target (partial relations - avoid these)
                        # So we maximize probability of NOT matching
                        total_log_prob -= np.log(p + 1e-10)
            
            return total_log_prob

        current_score = get_score(self.params)
        print(f"  [Transformer] Initial training score: {current_score:.4f}")
        
        for i in range(iterations):
            # Backup params
            backup = {k: v.copy() for k, v in self.params.items()}
            
            # Mutate
            self.mutate(rate=0.05)
            
            # Check new score
            new_score = get_score(self.params)
            
            if new_score > current_score:
                current_score = new_score
                # Keep mutation
            else:
                # Revert
                self.params = backup
                
        print(f"  [Transformer] Final training score: {current_score:.4f}")

    def mutate(self, rate=0.01):
        """Randomly mutate weights (Evolutionary Strategy)"""
        for k in self.params:
            if np.random.random() < 0.5:
                noise = np.random.normal(0, 0.1, self.params[k].shape)
                mask = (np.random.random(self.params[k].shape) < rate).astype(float)
                self.params[k] += noise * mask
