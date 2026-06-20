import os
import json
import numpy as np
import faiss
import jsonlines
import docx2txt
from sentence_transformers import SentenceTransformer

class CandidateVectorIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Initialization] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384 
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        self.id_map = []

    def prepare_text(self, candidate_obj):
        """
        Safely extracts and combines text components from a candidate object,
        ensuring no type mismatches (like None or dicts) crash the pipeline.
        """
        profile = candidate_obj.get("profile") or {}
        title = str(profile.get("title", "")).strip()
        summary = str(profile.get("summary", "")).strip()
        
        skills_data = candidate_obj.get("skills")
        skills_list = []
        
        if isinstance(skills_data, list):
            for skill in skills_data:
                if isinstance(skill, dict):
                    skills_list.append(str(skill.get("name", "")))
                elif skill:
                    skills_list.append(str(skill))
        elif isinstance(skills_data, str):
            skills_list.append(skills_data)
            
        skills_str = ", ".join([s for s in skills_list if s.strip()])
        
        text_segments = [
            f"Job Title: {title}.",
            f"Summary: {summary}.",
            f"Skills: {skills_str}."
        ]
        
        return " ".join([seg for seg in text_segments if seg.strip()])

    def build_index(self, jsonl_path, max_candidates=None):
        """
        Streams uncompressed JSONL profiles, generates embeddings, 
        and updates the local in-memory FAISS repository.
        """
        normalized_path = os.path.normpath(jsonl_path)
        print(f"[Processing] Generating embeddings from {normalized_path}...")
        
        embeddings_list = []
        
        with jsonlines.open(normalized_path) as reader:
            for idx, obj in enumerate(reader):
                if max_candidates and idx >= max_candidates:
                    break
                    
                candidate_id = obj.get("candidate_id") or obj.get("id")
                rich_text = self.prepare_text(obj)
                
                embedding = self.model.encode(rich_text, convert_to_numpy=True)
                
                faiss.normalize_L2(embedding.reshape(1, -1))
                
                embeddings_list.append(embedding)
                self.id_map.append(candidate_id)
                
                if (idx + 1) % 100 == 0:
                    print(f" -> Indexed {idx + 1} candidates...")
                    
        if embeddings_list:
            np_matrix = np.array(embeddings_list).astype('float32')
            self.index.add(np_matrix)
            print(f"[Success] FAISS In-Memory Index established with {self.index.ntotal} vectors.")
        else:
            print("[Error] Ingestion failed. No candidate vectors created.")

    def search(self, query_text, top_k=100):
        """
        Queries the FAISS index with a text string (like the Job Description)
        and returns the top_k closest matches.
        """
        query_vector = self.model.encode(query_text, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "candidate_id": self.id_map[idx],
                    "semantic_score": float(score)
                })
        return results

if __name__ == "__main__":
    print("=== Execution Trace: Phase 06 Vector Indexing ===")
    
    search_engine = CandidateVectorIndex()
    
    jd_path = "data/job_description.docx"
    print(f"[Processing] Extracting search targets from {jd_path}...")
    job_description_text = docx2txt.process(jd_path)
    
    candidates_path = "data/candidates.jsonl"
    
    if os.path.exists(candidates_path):
        search_engine.build_index(candidates_path, max_candidates=500)
        
        print("\n[Querying Engine] Executing vector match against Job Description criteria...")
        top_matches = search_engine.search(job_description_text, top_k=5)
        
        print("\n--- Top 5 Best-Fit Candidates Isolated via Vector Match ---")
        for rank, match in enumerate(top_matches, 1):
            print(f"Rank {rank}: ID = {match['candidate_id']} | Cosine Similarity = {match['semantic_score']:.4f}")
    else:
        print(f"[Error] Missing targeted file path: {candidates_path}")
