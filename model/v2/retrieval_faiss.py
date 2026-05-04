"""
STEP 5: RETRIEVAL (FAISS-based Dense Retrieval)
Uses FAISS to retrieve top-K candidate excipients based on context vector similarity.
"""

import torch
import torch.nn as nn
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None


class FAISSRetriever(nn.Module):
    """
    Dense retrieval using FAISS for efficient nearest neighbor search.
    
    Purpose:
        - Takes fused context vector (from step 4)
        - Performs efficient ANN search to retrieve top-K relevant excipients
        - Reduces computational load for downstream interaction modeling
    
    Workflow:
        1. Build FAISS index from excipient embeddings (once, offline)
        2. For each batch sample, search for top-K nearest excipients
        3. Return indices and similarity scores
    """
    
    def __init__(self, embedding_dim=256, use_gpu=True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0 if faiss else False
        self.index = None
        self.exc_embeddings = None
        
    def build_index(self, exc_embeddings):
        """
        Build FAISS index from excipient embeddings (offline step).
        
        Args:
            exc_embeddings: (V, embedding_dim) excipient embeddings from HGT
                           Can be numpy array or torch tensor
        """
        if isinstance(exc_embeddings, torch.Tensor):
            exc_embeddings = exc_embeddings.detach().cpu().numpy()
        
        exc_embeddings = np.ascontiguousarray(exc_embeddings.astype('float32'))
        
        # Create L2 distance index (Euclidean distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(exc_embeddings)
        self.exc_embeddings = exc_embeddings
        
        return self.index
    
    def forward(self, context_vectors, top_k=256):
        """
        Retrieve top-K excipients for each context vector.
        
        Args:
            context_vectors: (B, embedding_dim) fused context from step 4
            top_k: number of neighbors to retrieve (default 256)
            
        Returns:
            distances: (B, top_k) L2 distances (lower = more similar)
            indices: (B, top_k) indices of retrieved excipients in vocabulary
            scores: (B, top_k) similarity scores (higher = more similar)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")
        
        # Convert to numpy for FAISS
        if isinstance(context_vectors, torch.Tensor):
            queries = context_vectors.detach().cpu().numpy().astype('float32')
        else:
            queries = np.ascontiguousarray(context_vectors.astype('float32'))
        
        # Search
        distances, indices = self.index.search(queries, top_k)
        
        # Convert L2 distance to similarity score (inverse distance)
        # L2 distance: sqrt(sum((a-b)^2))
        # Similarity: 1 / (1 + distance) bounded to [0, 1]
        scores = 1.0 / (1.0 + distances)
        
        return distances, indices, scores


class HybridRetriever(nn.Module):
    """
    Hybrid retrieval combining:
    1. Dense retrieval (FAISS) - semantic similarity
    2. Sparse retrieval - TF-IDF or rule-based filtering
    
    Purpose:
        - Improve coverage for rare/niche excipients
        - Balance semantic relevance with diversity
    """
    
    def __init__(self, embedding_dim=256, use_gpu=True):
        super().__init__()
        
        self.dense_retriever = FAISSRetriever(embedding_dim, use_gpu)
        self.rule_filters = nn.ParameterDict()
        
    def build_index(self, exc_embeddings, sparse_features=None):
        """
        Build hybrid index.
        
        Args:
            exc_embeddings: (V, embedding_dim) from HGT
            sparse_features: Optional dict with sparse retrieval features
        """
        self.dense_retriever.build_index(exc_embeddings)
        
        if sparse_features:
            # Store sparse features for rule-based filtering
            for key, val in sparse_features.items():
                self.register_buffer(f'sparse_{key}', torch.from_numpy(val))
    
    def forward(self, context_vectors, top_k=256, alpha=0.7):
        """
        Hybrid retrieval: combine dense + sparse signals.
        
        Args:
            context_vectors: (B, embedding_dim)
            top_k: candidates to retrieve
            alpha: weight for dense retrieval (1-alpha for sparse)
            
        Returns:
            indices: (B, top_k) combined ranked indices
            scores: (B, top_k) combined scores
        """
        # Dense retrieval
        dense_dists, dense_indices, dense_scores = self.dense_retriever(
            context_vectors, top_k
        )
        
        # For now, return dense results; can integrate sparse later
        return dense_indices, dense_scores


class ContextAwareRetriever(nn.Module):
    """
    Multi-stage retrieval with context awareness.
    
    Stages:
        1. Coarse retrieval: Large top-K (e.g., 512) via FAISS
        2. Re-ranking: Learnable re-ranking using context + candidate interaction
    """
    
    def __init__(self, embedding_dim=256, top_k_coarse=512, top_k_final=256):
        super().__init__()
        
        self.coarse_retriever = FAISSRetriever(embedding_dim)
        self.top_k_coarse = top_k_coarse
        self.top_k_final = top_k_final
        
        # Re-ranker network
        self.reranker = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context_vectors, exc_embeddings):
        """
        Two-stage retrieval with re-ranking.
        
        Args:
            context_vectors: (B, embedding_dim)
            exc_embeddings: (V, embedding_dim) for re-ranking
            
        Returns:
            final_indices: (B, top_k_final)
            final_scores: (B, top_k_final)
        """
        # Stage 1: Coarse retrieval
        _, coarse_indices, coarse_scores = self.coarse_retriever(
            context_vectors, self.top_k_coarse
        )  # (B, top_k_coarse)
        
        B = context_vectors.shape[0]
        
        # Stage 2: Re-ranking
        final_indices = []
        final_scores = []
        
        for b in range(B):
            ctx = context_vectors[b:b+1]  # (1, embedding_dim)
            cand_indices = coarse_indices[b]  # (top_k_coarse,)
            cand_embs = exc_embeddings[cand_indices]  # (top_k_coarse, embedding_dim)
            
            # Compute re-ranking scores
            ctx_exp = ctx.expand(cand_embs.shape[0], -1)  # (top_k_coarse, embedding_dim)
            rerank_input = torch.cat([ctx_exp, cand_embs], dim=1)  # (top_k_coarse, 2*embedding_dim)
            rerank_scores = self.reranker(rerank_input).squeeze(1)  # (top_k_coarse,)
            
            # Select top-k final
            top_k_vals, top_k_idx = torch.topk(rerank_scores, min(self.top_k_final, len(rerank_scores)))
            
            final_idx = cand_indices[top_k_idx]  # Map back to original indices
            final_indices.append(final_idx)
            final_scores.append(top_k_vals)
        
        final_indices = torch.stack(final_indices)  # (B, top_k_final)
        final_scores = torch.stack(final_scores)    # (B, top_k_final)
        
        return final_indices, final_scores
