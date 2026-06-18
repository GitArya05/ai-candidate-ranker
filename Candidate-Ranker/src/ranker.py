import os
import json
import spacy
import pandas as pd

class CandidateRanker:
    def __init__(self):
        print("[Initialization] Loading spaCy NLP linguistic parser...")
        # Load the small english vocabulary model we downloaded in step 3
        self.nlp = spacy.load("en_core_web_sm")

    def extract_skills_programmatically(self, text):
        """Uses spaCy to extract nouns/proper nouns as fallback skill tokens."""
        doc = self.nlp(text.lower())
        # Filter tokens that look like technical skills or tools
        tokens = {token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 1}
        return tokens

    def calculate_hard_constraints(self, candidate_obj, jd_text):
        """
        Computes an explicit technical overlap score by checking 
        skills listed in candidate json vs keywords in the JD.
        """
        # Parse job description tokens
        jd_tokens = self.extract_skills_programmatically(jd_text)
        
        # Parse candidate skills
        candidate_skills = candidate_obj.get("skills", [])
        if isinstance(candidate_skills, list):
            cand_skills_set = {str(s).lower() for s in candidate_skills}
        else:
            cand_skills_set = self.extract_skills_programmatically(str(candidate_skills))
            
        if not jd_tokens or not cand_skills_set:
            return 0.0
            
        # Calculate intersection ratio
        matching_skills = cand_skills_set.intersection(jd_tokens)
        return float(len(matching_skills) / max(len(cand_skills_set), 1))

    def calculate_behavioral_signals(self, candidate_obj):
        """
        Parses attributes from the 'redrob_signals' object block.
        Applies point metrics based on candidate engagement.
        """
        score = 0.0
        signals = candidate_obj.get("redrob_signals", {}) or {}
        
        # Example metric weights based on standard behavioral evaluations:
        # 1. Profile completeness/update health
        if str(signals.get("profile_updated", "")).lower() == "true":
            score += 0.4
            
        # 2. Activity/Responsiveness metrics
        responsiveness = signals.get("responsiveness_score", 0)
        if isinstance(responsiveness, (int, float)):
            # Normalize score out of 100 to a max value of 0.4
            score += (responsiveness / 100.0) * 0.4
            
        # 3. Placement velocity / Actively looking status
        if str(signals.get("actively_looking", "")).lower() == "true":
            score += 0.2
            
        return min(score, 1.0) # Cap baseline total at 1.0

    def compute_final_rankings(self, top_semantic_matches, all_candidates_map, jd_text):
        """
        Combines Stage 1 Semantic Scores with Stage 2 Hard Constraints & Behavioral Metrics
        to compile the final, unified ranked leaderboard.
        """
        final_rows = []
        
        # Configurable mathematical scoring weights
        w_sem = 0.5   # 50% weight for conceptual/semantic match
        w_con = 0.3   # 30% weight for explicit tool/skill matching
        w_beh = 0.2   # 20% weight for engagement metrics
        
        print(f"[Processing] Re-ranking top {len(top_semantic_matches)} filtered candidates...")
        
        for item in top_semantic_matches:
            c_id = item["candidate_id"]
            semantic_score = item["semantic_score"]
            
            # Fetch the complete raw JSON object from our map cache
            candidate_obj = all_candidates_map.get(c_id)
            if not candidate_obj:
                continue
                
            # Run Stage 2 metric evaluations
            constraint_score = self.calculate_hard_constraints(candidate_obj, jd_text)
            behavioral_score = self.calculate_behavioral_signals(candidate_obj)
            
            # Unified Multi-Dimensional scoring calculation
            final_score = (w_sem * semantic_score) + (w_con * constraint_score) + (w_beh * behavioral_score)
            
            final_rows.append({
                "candidate_id": c_id,
                "final_score": final_score,
                "semantic_component": semantic_score,
                "technical_component": constraint_score,
                "behavioral_component": behavioral_score
            })
            
        # Convert to Pandas DataFrame for high-performance sorting
        df = pd.DataFrame(final_rows)
        if not df.empty:
            df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
            # Add absolute Rank coordinate column (starting at 1)
            df["rank"] = df.index + 1
        return df

if __name__ == "__main__":
    print("=== Execution Trace: Phase 07/08 Feature Ranker ===")
    ranker = CandidateRanker()
    
    # Mock data setup to test the script components independently
    mock_candidate = {
        "candidate_id": "CAND_0000389",
        "skills": ["Python", "Machine Learning", "PyTorch"],
        "redrob_signals": {"profile_updated": "true", "responsiveness_score": 85, "actively_looking": "true"}
    }
    mock_jd = "Seeking a Machine Learning Engineer proficient in Python, PyTorch, and NLP models."
    
    c_score = ranker.calculate_hard_constraints(mock_candidate, mock_jd)
    b_score = ranker.calculate_behavioral_signals(mock_candidate)
    
    print(f"\n[Sanity Check Results]")
    print(f" -> Computed Technical Skills Overlap Score: {c_score:.4f}")
    print(f" -> Computed Behavioral Engagement Score: {b_score:.4f}")