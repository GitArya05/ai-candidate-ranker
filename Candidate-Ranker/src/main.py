import os
import docx2txt
import jsonlines
from vector_search import CandidateVectorIndex
from ranker import CandidateRanker

def run_end_to_end_pipeline():
    print("==================================================")
    print("STARTING THE INTELLIGENT CANDIDATE DISCOVERY ENGINE")
    print("==================================================")

    # Define absolute input and output resource paths
    jd_path = os.path.normpath("data/job_description.docx")
    candidates_path = os.path.normpath("data/candidates.jsonl")
    output_path = os.path.normpath("data/submission.csv")

    # 1. Parse and extract target criteria
    print("\n[Step 1/5] Extracting Job Description text...")
    if not os.path.exists(jd_path):
        print(f"[Critical Error] Missing job description at {jd_path}")
        return
    jd_text = docx2txt.process(jd_path)

    # 2. Build In-Memory Database Map Cache
    print("\n[Step 2/5] Creating primary streaming candidate cache matrix...")
    if not os.path.exists(candidates_path):
        print(f"[Critical Error] Missing dataset at {candidates_path}")
        return
        
    all_candidates_map = {}
    with jsonlines.open(candidates_path) as reader:
        for obj in reader:
            c_id = obj.get("candidate_id") or obj.get("id")
            if c_id:
                all_candidates_map[c_id] = obj

    total_pool_size = len(all_candidates_map)
    print(f"[Success] Cached {total_pool_size} candidate profiles safely in memory.")

    # 3. Execute Stage 1 Fast Retrieval (FAISS)
    print("\n[Step 3/5] Initializing FAISS Vector Search Index...")
    vector_engine = CandidateVectorIndex()
    # Build index using our cached values directly from file path
    vector_engine.build_index(candidates_path)
    
    # Isolate the top 100 matching candidates semantically
    # If the candidate pool is smaller than 100, scale dynamically to its max size
    target_k = min(100, total_pool_size)
    print(f"\n[Filtering] Isolating Top {target_k} semantic candidates matching the JD...")
    top_semantic_matches = vector_engine.search(jd_text, top_k=target_k)

    # 4. Execute Stage 2 Precision Feature Re-ranking
    print("\n[Step 4/5] Running fine-grained feature re-ranking score matrix...")
    ranker = CandidateRanker()
    final_leaderboard_df = ranker.compute_final_rankings(
        top_semantic_matches=top_semantic_matches,
        all_candidates_map=all_candidates_map,
        jd_text=jd_text
    )

    # 5. Export Final Sorted Results
    print("\n[Step 5/5] Structuring submission file output formatting...")
    if not final_leaderboard_df.empty:
        # Reformat columns to align perfectly with sample_submission.csv format
        # Common hackathon targets look for: candidate_id, rank, final_score
        submission_df = final_leaderboard_df[["candidate_id", "rank", "final_score"]]
        
        # Save output
        submission_df.to_csv(output_path, index=False)
        print(f"[Success] Pipeline completed! Shortlist saved to: {output_path}")
        print(f"Top 3 Shortlisted Profiles:")
        print(submission_df.head(3).to_string(index=False))
    else:
        print("[Error] Final leaderboard generation resulted in an empty configuration.")

if __name__ == "__main__":
    run_end_to_end_pipeline()