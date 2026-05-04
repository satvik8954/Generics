"""
STEP 8: BEAM SEARCH — Combination Generation

Purpose: Search through excipient combinations to find promising sets that work well together.

**Beam Search Strategy**:
  1. Start with top-K excipients from retrieval
  2. Iteratively expand: try adding each excipient to current beams
  3. Score each combination (combination score + interaction bonus)
  4. Keep top-B beams at each step
  5. Output: Top-N ranked combinations

**Why Beam Search?**
  - Excipient compatibility is non-additive (synergistic/antagonistic effects)
  - Greedy approach may miss better multi-excipient sets
  - Beam search balances quality (top beams) vs speed (limited beam width)

**Future**: Can replace with GFlowNet for learnable exploration once model stabilizes.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from itertools import combinations


class ExcipientCombinationScorer(nn.Module):
    """
    Scores combinations of excipients using learned pairwise interactions.
    
    Combination Score = Base Score + Interaction Bonus
    where:
      - Base Score = mean/sum of individual excipient scores
      - Interaction Bonus = learned interactions between pairs (negative = incompatibility)
    """
    
    def __init__(self, hidden_dim: int = 256, num_excipients: int = 1299):
        """
        Args:
            hidden_dim: Dimension of excipient embeddings
            num_excipients: Total number of unique excipients in vocab
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_excipients = num_excipients
        
        # Excipient embeddings (initialized, filled during inference)
        self.excipient_embeddings = nn.Embedding(num_excipients, hidden_dim)
        
        # Pairwise interaction network
        # Input: concat of two excipient embeddings
        # Output: scalar interaction score
        self.interaction_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single scalar output
        )
        
        # Combination aggregator (learns how to weight individual + pairwise scores)
        self.aggregator = nn.Sequential(
            nn.Linear(2, 16),  # (individual_score, interaction_avg)
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def score_individual(self, excipient_scores: torch.Tensor) -> torch.Tensor:
        """Average individual excipient scores."""
        return excipient_scores.mean(dim=-1, keepdim=True)
    
    def score_pairwise_interactions(self, excipient_indices: torch.Tensor) -> torch.Tensor:
        """
        Score pairwise interactions in a combination.
        
        Args:
            excipient_indices: (set_size,) indices of excipients in combination
            
        Returns:
            Scalar: average pairwise interaction score
        """
        if excipient_indices.shape[0] < 2:
            return torch.tensor(0.0, device=excipient_indices.device)
        
        # Get embeddings
        embs = self.excipient_embeddings(excipient_indices)  # (set_size, hidden_dim)
        
        # Compute all pairwise interactions
        interactions = []
        for i in range(len(excipient_indices)):
            for j in range(i + 1, len(excipient_indices)):
                # Concatenate pair embeddings
                pair = torch.cat([embs[i], embs[j]], dim=0)  # (2*hidden_dim,)
                interaction_score = self.interaction_network(pair)  # (1,)
                interactions.append(interaction_score)
        
        if len(interactions) == 0:
            return torch.tensor(0.0, device=excipient_indices.device)
        
        # Average pairwise interactions
        avg_interaction = torch.stack(interactions).mean()
        return avg_interaction
    
    def forward(self, excipient_indices: torch.Tensor, 
                individual_scores: torch.Tensor) -> torch.Tensor:
        """
        Score a combination of excipients.
        
        Args:
            excipient_indices: (set_size,) indices in excipient vocab
            individual_scores: (set_size,) retrieval scores for each excipient
            
        Returns:
            Scalar: combination score in [0, 1] range
        """
        # Individual component
        individual_avg = individual_scores.mean()
        
        # Pairwise interaction component
        interaction_avg = self.score_pairwise_interactions(excipient_indices)
        
        # Combine
        combined = torch.stack([individual_avg, interaction_avg], dim=0).unsqueeze(0)  # (1, 2)
        score = torch.sigmoid(self.aggregator(combined).squeeze())
        
        return score


class BeamSearchCombinationGenerator:
    """
    Beam search for finding promising excipient combinations.
    
    Algorithm:
      1. Start with top-1 excipient (best from retrieval)
      2. For each step:
         - For each beam, try adding each candidate excipient
         - Score resulting combination
         - Keep top-B combinations
      3. Stop after max_steps or when no improvement
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        max_combination_size: int = 5,
        scorer: Optional[ExcipientCombinationScorer] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            num_beams: Number of beams to maintain (beam width)
            max_combination_size: Maximum excipients per combination
            scorer: ExcipientCombinationScorer instance (if None, uses naive scoring)
            device: Device to run on
        """
        self.num_beams = num_beams
        self.max_combination_size = max_combination_size
        self.scorer = scorer
        self.device = device
    
    def score_combination_naive(
        self,
        excipient_indices: List[int],
        individual_scores: torch.Tensor
    ) -> float:
        """
        Naive scoring: mean of individual scores + small penalty for diversity.
        (Used when no learned scorer is available)
        """
        indices_set = set(excipient_indices)
        individual_avg = individual_scores[[idx for idx in excipient_indices]].mean().item()
        
        # Slight penalty for large combinations (encourage compactness)
        diversity_penalty = len(excipient_indices) * 0.02
        
        return individual_avg - diversity_penalty
    
    def search(
        self,
        top_k_indices: torch.Tensor,
        top_k_scores: torch.Tensor,
        all_excipient_scores: torch.Tensor
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Perform beam search to find promising combinations.
        
        Args:
            top_k_indices: (k,) indices of top-K excipients from retrieval
            top_k_scores: (k,) scores of top-K excipients
            all_excipient_scores: (num_excipients,) all excipient scores
            
        Returns:
            combinations: List of combinations (each is list of indices)
            combination_scores: List of scores for each combination
        """
        k = len(top_k_indices)
        
        # Initialize beams with single excipients
        beams = [
            {
                'combination': [int(top_k_indices[i].item())],
                'score': float(top_k_scores[i].item())
            }
            for i in range(min(self.num_beams, k))
        ]
        
        # Iteratively expand combinations
        for step in range(1, self.max_combination_size):
            candidates = []
            
            for beam in beams:
                current_combo = set(beam['combination'])
                
                # Try adding each unused excipient
                for candidate_idx in top_k_indices:
                    idx = int(candidate_idx.item())
                    
                    if idx not in current_combo:
                        new_combo = beam['combination'] + [idx]
                        
                        # Score new combination
                        if self.scorer is not None:
                            combo_tensor = torch.tensor(new_combo, device=self.device)
                            scores_tensor = all_excipient_scores[new_combo]
                            new_score = self.scorer(combo_tensor, scores_tensor).item()
                        else:
                            new_score = self.score_combination_naive(
                                new_combo,
                                all_excipient_scores
                            )
                        
                        candidates.append({
                            'combination': new_combo,
                            'score': new_score
                        })
            
            if not candidates:
                break  # No more candidates
            
            # Sort and keep top beams
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:self.num_beams]
            
            # Early stopping: if score doesn't improve, break
            if step > 1 and beams[0]['score'] <= candidates[0]['score'] * 0.99:
                break
        
        # Sort final beams by score
        beams.sort(key=lambda x: x['score'], reverse=True)
        
        combinations = [beam['combination'] for beam in beams]
        scores = [beam['score'] for beam in beams]
        
        return combinations, scores


class SearchStage2(nn.Module):
    """
    Complete Stage 2 Search: takes top-K excipients and outputs promising combinations.
    
    Input:
      - top_k_indices: (batch_size, k) top-K excipient indices from retrieval
      - top_k_scores: (batch_size, k) scores from retrieval
      - all_excipient_scores: (batch_size, num_excipients) scores for all excipients
      
    Output:
      - combinations: (batch_size, num_beams) list of promising combinations
      - combination_scores: (batch_size, num_beams) scores for each combination
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        max_combination_size: int = 5,
        use_learned_scorer: bool = False,
        hidden_dim: int = 256,
        num_excipients: int = 1299,
        device: str = 'cpu'
    ):
        super().__init__()
        self.num_beams = num_beams
        self.max_combination_size = max_combination_size
        self.device = device
        
        # Optional learned scorer
        if use_learned_scorer:
            self.scorer = ExcipientCombinationScorer(hidden_dim, num_excipients)
        else:
            self.scorer = None
        
        self.beam_search = BeamSearchCombinationGenerator(
            num_beams=num_beams,
            max_combination_size=max_combination_size,
            scorer=self.scorer,
            device=device
        )
    
    def forward(
        self,
        top_k_indices: torch.Tensor,
        top_k_scores: torch.Tensor,
        all_excipient_scores: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Args:
            top_k_indices: (batch_size, k) indices of top-K excipients
            top_k_scores: (batch_size, k) retrieval scores
            all_excipient_scores: (batch_size, num_excipients) or (num_excipients,)
            
        Returns:
            combinations: List[List[List[int]]] - combinations per batch
            scores: (batch_size, num_beams) tensor of combination scores
        """
        batch_size = top_k_indices.shape[0]
        
        combinations_batch = []
        scores_batch = []
        
        for b in range(batch_size):
            combinations, scores = self.beam_search.search(
                top_k_indices=top_k_indices[b],
                top_k_scores=top_k_scores[b],
                all_excipient_scores=all_excipient_scores[b] if all_excipient_scores.dim() > 1
                                     else all_excipient_scores
            )
            
            combinations_batch.append(combinations)
            
            # Pad scores to max_beams
            padded_scores = scores + [0.0] * (self.num_beams - len(scores))
            scores_batch.append(padded_scores[:self.num_beams])
        
        scores_tensor = torch.tensor(scores_batch, device=self.device, dtype=torch.float32)
        
        return combinations_batch, scores_tensor


# ============================================================================
# FUTURE: GFlowNet-based search (placeholder for later integration)
# ============================================================================

class GFlowNetSearchPlaceholder(nn.Module):
    """
    Placeholder for GFlowNet-based combination search.
    
    To be implemented when:
      1. Model is more optimized
      2. We have stable training signals
      3. We want to learn exploration policy instead of fixed beam search
    
    GFlowNet advantages:
      - Learns to explore the combination space efficiently
      - Can handle larger combination sizes
      - Naturally handles multi-modal solution sets
    """
    
    def __init__(self):
        super().__init__()
        self.note = "GFlowNet integration: Coming soon when model stabilizes"
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("GFlowNet search: Coming soon. Use BeamSearch for now.")
