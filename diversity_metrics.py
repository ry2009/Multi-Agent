from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class DiversityMetrics:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
    def semantic_diversity(self, responses: List[str]) -> float:
        """Calculate semantic diversity using sentence embeddings."""
        if len(responses) < 2:
            return 0.0
            
        # Get embeddings for all responses
        embeddings = self.embedding_model.encode(responses)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Convert to diversity score (1 - average similarity)
        diversity = 1 - (np.sum(similarities) - len(responses)) / (len(responses) * (len(responses) - 1))
        return float(diversity)
    
    def cluster_diversity(self, responses: List[str], n_clusters: int = None) -> Tuple[float, List[int]]:
        """Measure diversity by clustering responses and analyzing cluster distribution."""
        if len(responses) < 2:
            return 0.0, [0]
            
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(len(responses) // 2 + 1, len(responses))
            
        # Get embeddings
        embeddings = self.embedding_model.encode(responses)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate cluster distribution entropy
        cluster_counts = np.bincount(clusters)
        probabilities = cluster_counts / len(responses)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log2(n_clusters)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy), clusters.tolist()
    
    def lexical_diversity(self, responses: List[str]) -> float:
        """Calculate lexical diversity based on vocabulary usage."""
        if not responses:
            return 0.0
            
        # Get unique words across all responses
        all_words = set()
        response_words = []
        
        for response in responses:
            words = set(response.lower().split())
            all_words.update(words)
            response_words.append(words)
        
        # Calculate average Jaccard distance between responses
        n = len(responses)
        if n < 2:
            return 0.0
            
        total_distance = 0.0
        comparisons = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(response_words[i] & response_words[j])
                union = len(response_words[i] | response_words[j])
                if union > 0:
                    distance = 1 - (intersection / union)
                    total_distance += distance
                    comparisons += 1
        
        return float(total_distance / comparisons) if comparisons > 0 else 0.0
    
    def get_comprehensive_diversity(self, responses: List[str]) -> dict:
        """Calculate comprehensive diversity metrics."""
        return {
            'semantic': self.semantic_diversity(responses),
            'lexical': self.lexical_diversity(responses),
            'cluster': self.cluster_diversity(responses)[0]
        }
    
    def get_diversity_score(self, responses: List[str], weights: dict = None) -> float:
        """Get weighted diversity score."""
        if weights is None:
            weights = {
                'semantic': 0.4,
                'lexical': 0.3,
                'cluster': 0.3
            }
            
        metrics = self.get_comprehensive_diversity(responses)
        return sum(metrics[key] * weight for key, weight in weights.items()) 