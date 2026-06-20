import os
import json
import docx2txt
import jsonlines
from jsonschema import validate, ValidationError

def parse_docx(file_path):
    """Safely extracts clean string text from a Windows-based .docx file."""
    normalized_path = os.path.normpath(file_path)
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Target document missing at: {normalized_path}")
    return docx2txt.process(normalized_path)

def stream_and_validate_candidates(jsonl_path, schema_path=None):
    """
    Streams candidates using jsonlines to protect system memory.
    Validates structure against candidate_schema.json if provided.
    """
    normalized_jsonl = os.path.normpath(jsonl_path)
    if not os.path.exists(normalized_jsonl):
        raise FileNotFoundError(f"Dataset target missing at: {normalized_jsonl}")
        
    schema = None
    if schema_path and os.path.exists(os.path.normpath(schema_path)):
        with open(os.path.normpath(schema_path), 'r', encoding='utf-8') as sf:
            schema = json.load(sf)
            
    with jsonlines.open(normalized_jsonl) as reader:
        for idx, obj in enumerate(reader):
            if schema:
                try:
                    validate(instance=obj, schema=schema)
                except ValidationError as e:
                    print(f"[Warning] Row {idx} failed structural schema evaluation: {e.message}")
                    continue 
            yield obj

if __name__ == "__main__":
    print("=== Execution Trace: Phase 05 Core Ingestion ===")
    
    jd_raw = parse_docx("data/job_description.docx")
    print(f"[Success] Parsed Job Description. Total character length: {len(jd_raw)}")
    
    print("\n[Processing] Initializing JSONL stream validation...")
    streamer = stream_and_validate_candidates(
        jsonl_path="data/candidates.jsonl", 
        schema_path="data/candidate_schema.json"
    )
    
    try:
        sample_candidate = next(streamer)
        print(f"[Success] Stream engine verified! Parsed Candidate ID: {sample_candidate.get('id')}")
        print("Available Top-Level Object Keys:", list(sample_candidate.keys()))
    except StopIteration:
        print("[Error] Ingestion engine reached EOF without capturing valid JSON structures.")
