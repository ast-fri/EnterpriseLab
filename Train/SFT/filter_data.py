import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm
import argparse

class SemanticDeduplicator:
    """
    Deduplicates tasks based on semantic similarity of user messages.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.85):
        """
        Initialize the deduplicator.
        
        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Cosine similarity threshold (0-1). 
                                 Tasks with similarity >= threshold are considered duplicates.
                                 Recommended: 0.80-0.90 for near-duplicates
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        print(f"Similarity threshold: {similarity_threshold}")
        
    def extract_user_content(self, task: Dict) -> str:
        """
        Extract user content from messages in a task.
        
        Args:
            task: Dictionary containing 'messages' key
            
        Returns:
            User message content as string
        """
        # messages = task.get('messages', [])
        return task.get('query', '')
        
    
    def deduplicate(self, input_file: str, output_file: str = None) -> List[Dict]:
        """
        Deduplicate tasks from JSON file based on semantic similarity.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to save deduplicated JSON (optional)
            
        Returns:
            List of unique tasks
        """
        # Load data
        print(f"\n📂 Loading data from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            all_tasks = json.load(f)
        
        print(f"Total tasks loaded: {len(all_tasks)}")
        
        # Extract user contents
        print("\n🔍 Extracting user messages...")
        user_contents = [self.extract_user_content(task) for task in all_tasks]
        
        # Filter out empty contents
        valid_indices = [i for i, content in enumerate(user_contents) if content.strip()]
        all_tasks = [all_tasks[i] for i in valid_indices]
        user_contents = [user_contents[i] for i in valid_indices]
        
        print(f"Valid tasks with user messages: {len(all_tasks)}")
        
        if len(all_tasks) == 0:
            print("⚠️ No valid tasks found!")
            return []
        
        # Initialize unique tasks list
        unique_tasks = []
        unique_embeddings = []
        duplicate_count = 0
        
        print(f"\n🚀 Computing embeddings and checking for duplicates...")
        print(f"Threshold: {self.threshold} (tasks with similarity >= {self.threshold} are considered duplicates)")
        
        # Process each task
        for idx, (task, user_content) in enumerate(tqdm(zip(all_tasks, user_contents), total=len(all_tasks))):
            # Compute embedding for current task
            current_embedding = self.model.encode(user_content, convert_to_tensor=False)
            
            # Check similarity with already added unique tasks
            is_duplicate = False
            max_similarity = 0.0
            
            if len(unique_embeddings) > 0:
                # Compute cosine similarity with all unique embeddings
                similarities = self.model.similarity(
                    current_embedding, 
                    np.array(unique_embeddings)
                ).numpy().flatten()
                
                max_similarity = similarities.max()
                
                # Check if any similarity exceeds threshold
                if max_similarity >= self.threshold:
                    is_duplicate = True
                    duplicate_count += 1
            
            # Add to unique tasks if not duplicate
            if not is_duplicate:
                unique_tasks.append(task)
                unique_embeddings.append(current_embedding)
        
        print(f"\n✅ Deduplication complete!")
        print(f"Original tasks: {len(all_tasks)}")
        print(f"Unique tasks: {len(unique_tasks)}")
        print(f"Duplicates removed: {duplicate_count}")
        print(f"Reduction: {(duplicate_count/len(all_tasks))*100:.2f}%")
        
        # Save to file if output path provided
        if output_file:
            print(f"\n💾 Saving deduplicated data to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_tasks, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(unique_tasks)} unique tasks")
        
        return unique_tasks
    
    def find_similar_pairs(self, input_file: str, top_k: int = 10) -> List[tuple]:
        """
        Find the most similar task pairs (for analysis).
        
        Args:
            input_file: Path to input JSON file
            top_k: Number of top similar pairs to return
            
        Returns:
            List of (task1_idx, task2_idx, similarity_score) tuples
        """
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            all_tasks = json.load(f)
        
        user_contents = [self.extract_user_content(task) for task in all_tasks]
        
        print(f"Computing embeddings for {len(user_contents)} tasks...")
        embeddings = self.model.encode(user_contents, show_progress_bar=True)
        
        print("Computing similarity matrix...")
        similarity_matrix = self.model.similarity(embeddings, embeddings).numpy()
        
        # Set diagonal to -1 (ignore self-similarity)
        np.fill_diagonal(similarity_matrix, -1)
        
        # Find top-k most similar pairs
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                similar_pairs.append((i, j, similarity_matrix[i][j]))
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop {top_k} most similar task pairs:")
        for idx, (i, j, score) in enumerate(similar_pairs[:top_k], 1):
            print(f"\n{idx}. Similarity: {score:.4f}")
            print(f"   Task {i}: {user_contents[i][:100]}...")
            print(f"   Task {j}: {user_contents[j][:100]}...")
        
        return similar_pairs[:top_k]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and deduplicate training data")
    parser.add_argument('--input_path', required=True, help='Path to input raw data JSON')
    parser.add_argument('--output_path', required=True, help='Path to save filtered data JSON')
    parser.add_argument('--min_quality_score', type=float, default=0.7, help='Minimum quality score (similarity threshold)')
    
    args = parser.parse_args()
    
    deduplicator = SemanticDeduplicator(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=args.min_quality_score
    )
    
    unique = deduplicator.deduplicate(
        input_file=args.input_path,
        output_file=args.output_path
    )
