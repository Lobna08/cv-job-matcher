"""
CV-to-Job Matching System using Gemini Embeddings and Supabase
Production-ready implementation for semantic similarity matching
"""

import os
from typing import List, Dict, Tuple, Optional
# import google.generativeai as genai
from google import genai  # New import
from google.genai import types  # For EmbedContentConfig
from supabase import create_client, Client
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Import text preparation functions
from text_preparation import (
    prepare_job_announcement_text,
    prepare_cv_text,
    prepare_job_content_from_announcement,
    prepare_cv_content_from_user_cv
)

# Module-level placeholder for Gemini client so `init_clients` can reuse it
# Initialized to None to avoid NameError when checked inside the function.
gemini_client: Optional[genai.Client] = None



# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Gemini embedding model configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

# Batch processing configuration
BATCH_SIZE = 100  # Process embeddings in batches
RATE_LIMIT_DELAY = 0.1  # Seconds between API calls


# ============================================================================
# Initialize clients
# ============================================================================





def init_clients() -> Tuple[Client, genai.Client]:
    global gemini_client

    if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
        raise ValueError("Missing environment variables")
    
    # Initialize NEW Gemini Client
    if gemini_client is None:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Initialize Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return supabase, gemini_client


# ============================================================================
# Embedding Generation
# ============================================================================

def generate_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embedding using Gemini gemini-embedding-001.
    
    Args:
        text: Input text to embed
        task_type: Task type for Gemini API
            - "RETRIEVAL_DOCUMENT": For documents to be retrieved (jobs, CVs when storing)
            - "RETRIEVAL_QUERY": For queries (CV when matching to jobs)
    
    Returns:
        List of floats representing the embedding vector (768 dimensions)
    
    Why Gemini gemini-embedding-001:
        - Optimized for semantic similarity tasks
        - 768-dimensional vectors balance quality and storage
        - Task-type parameter optimizes embeddings for retrieval vs query context
    """
    try:
        _, client = init_clients()

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSION)
        )

        # New SDK response structure
        return response.embeddings[0].values

    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")


def generate_embeddings_batch(
    texts: List[str], 
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts with rate limiting.
    
    Args:
        texts: List of texts to embed
        task_type: Task type for Gemini API
    
    Returns:
        List of embedding vectors (None for failed embeddings)
    """
    embeddings = []
    
    for i, text in enumerate(texts):
        try:
            embedding = generate_embedding(text, task_type)
            embeddings.append(embedding)
            
            # Rate limiting to avoid hitting API limits
            if i < len(texts) - 1:
                time.sleep(RATE_LIMIT_DELAY)
                
        except Exception as e:
            print(f"Error generating embedding for text {i}: {str(e)}")
            # Store None for failed embeddings
            embeddings.append(None)
    
    return embeddings


# ============================================================================
# Job Announcement Processing
# ============================================================================

def populate_jobs_from_announcements(supabase: Client) -> Dict[str, int]:
    """
    Populate jobs table with content from job_announcements.
    Creates entries in jobs table with prepared text content.
    
    Args:
        supabase: Supabase client
    
    Returns:
        Dictionary with processing statistics
    """
    print("Populating jobs table from job_announcements...")
    
    # Fetch all job announcements
    response = supabase.table("job_announcements").select("*").execute()
    announcements = response.data
    
    if not announcements:
        print("No job announcements found.")
        return {"total": 0, "created": 0, "skipped": 0}
    
    print(f"Found {len(announcements)} job announcements.")
    
    total = len(announcements)
    created = 0
    skipped = 0
    
    for announcement in announcements:
        try:
            # Check if job already exists
            existing = supabase.table("jobs").select("id").eq(
                "job_announcement_id", announcement["id"]
            ).execute()
            
            if existing.data:
                skipped += 1
                continue
            
            # Prepare content text
            content = prepare_job_content_from_announcement(announcement)
            
            # Insert into jobs table
            supabase.table("jobs").insert({
                "job_announcement_id": announcement["id"],
                "content": content,
                "embedding": None  # Will be populated by embed_all_jobs
            }).execute()
            
            created += 1
            
        except Exception as e:
            print(f"Failed to process announcement {announcement['id']}: {str(e)}")
            skipped += 1
    
    print(f"Population complete. Total: {total}, Created: {created}, Skipped: {skipped}")
    return {"total": total, "created": created, "skipped": skipped}


def embed_all_jobs(supabase: Client, batch_size: int = BATCH_SIZE) -> Dict[str, int]:
    """
    Generate and store embeddings for all jobs in the database.
    
    This function:
    1. Fetches all jobs without embeddings
    2. Generates embeddings using Gemini text-embedding-004
    3. Stores embeddings back in the database
    
    Args:
        supabase: Supabase client
        batch_size: Number of records to process at once
    
    Returns:
        Dictionary with processing statistics
    
    Why generate embeddings offline:
        - Pre-computation enables fast real-time matching
        - Embeddings are static until content changes
        - Batch processing is more efficient than on-demand generation
        - Reduces API costs and latency for user queries
    """
    print("Starting job embedding pipeline...")
    
    # Fetch all jobs without embeddings
    response = supabase.table("jobs").select("id, content").is_("embedding", "null").execute()
    jobs = response.data
    
    if not jobs:
        print("No jobs found without embeddings.")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"Found {len(jobs)} jobs to embed.")
    
    total = len(jobs)
    success = 0
    failed = 0
    
    # Process in batches
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(jobs) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        # Generate embeddings for batch
        texts = [job["content"] for job in batch]
        embeddings = generate_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        
        # Update database
        for job, embedding in zip(batch, embeddings):
            if embedding is not None:
                try:
                    supabase.table("jobs").update({
                        "embedding": embedding
                    }).eq("id", job["id"]).execute()
                    success += 1
                except Exception as e:
                    print(f"Failed to update job {job['id']}: {str(e)}")
                    failed += 1
            else:
                failed += 1
        
        print(f"Batch {batch_num} complete. Success: {success}, Failed: {failed}")
    
    print(f"\nJob embedding complete. Total: {total}, Success: {success}, Failed: {failed}")
    return {"total": total, "success": success, "failed": failed}


# ============================================================================
# User CV Processing
# ============================================================================

def populate_cvs_from_user_cvs(supabase: Client) -> Dict[str, int]:
    """
    Populate cvs table with content from user_cvs.
    Creates entries in cvs table with prepared text content.
    
    Args:
        supabase: Supabase client
    
    Returns:
        Dictionary with processing statistics
    """
    print("Populating cvs table from user_cvs...")
    
    # Fetch all user CVs
    response = supabase.table("user_cvs").select("*").execute()
    user_cvs = response.data
    
    if not user_cvs:
        print("No user CVs found.")
        return {"total": 0, "created": 0, "skipped": 0}
    
    print(f"Found {len(user_cvs)} user CVs.")
    
    total = len(user_cvs)
    created = 0
    skipped = 0
    
    for user_cv in user_cvs:
        try:
            # Check if CV already exists
            existing = supabase.table("cvs").select("id").eq(
                "user_cv_id", user_cv["id"]
            ).execute()
            
            if existing.data:
                skipped += 1
                continue
            
            # Prepare content text
            content = prepare_cv_content_from_user_cv(user_cv)
            
            # Insert into cvs table
            supabase.table("cvs").insert({
                "user_cv_id": user_cv["id"],
                "content": content,
                "embedding": None  # Will be populated by embed_all_cvs
            }).execute()
            
            created += 1
            
        except Exception as e:
            print(f"Failed to process user CV {user_cv['id']}: {str(e)}")
            skipped += 1
    
    print(f"Population complete. Total: {total}, Created: {created}, Skipped: {skipped}")
    return {"total": total, "created": created, "skipped": skipped}


def embed_all_cvs(supabase: Client, batch_size: int = BATCH_SIZE) -> Dict[str, int]:
    """
    Generate and store embeddings for all CVs in the database.
    
    Args:
        supabase: Supabase client
        batch_size: Number of records to process at once
    
    Returns:
        Dictionary with processing statistics
    """
    print("Starting CV embedding pipeline...")
    
    # Fetch all CVs without embeddings
    response = supabase.table("cvs").select("id, content").is_("embedding", "null").execute()
    cvs = response.data
    
    if not cvs:
        print("No CVs found without embeddings.")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"Found {len(cvs)} CVs to embed.")
    
    total = len(cvs)
    success = 0
    failed = 0
    
    # Process in batches
    for i in range(0, len(cvs), batch_size):
        batch = cvs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(cvs) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        # Generate embeddings for batch
        texts = [cv["content"] for cv in batch]
        embeddings = generate_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        
        # Update database
        for cv, embedding in zip(batch, embeddings):
            if embedding is not None:
                try:
                    supabase.table("cvs").update({
                        "embedding": embedding
                    }).eq("id", cv["id"]).execute()
                    success += 1
                except Exception as e:
                    print(f"Failed to update CV {cv['id']}: {str(e)}")
                    failed += 1
            else:
                failed += 1
        
        print(f"Batch {batch_num} complete. Success: {success}, Failed: {failed}")
    
    print(f"\nCV embedding complete. Total: {total}, Success: {success}, Failed: {failed}")
    return {"total": total, "success": success, "failed": failed}


# ============================================================================
# Job Matching
# ============================================================================

def get_top_jobs_for_cv(
    supabase: Client,
    cv_id: str,
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Find top-K most similar jobs for a given CV using pgvector.
    
    How the matching query works:
    1. Retrieves the CV's pre-computed embedding from the database
    2. Uses PostgreSQL's pgvector extension with cosine distance operator (<=>)
    3. HNSW index accelerates approximate nearest neighbor search
    4. Returns jobs sorted by similarity (1 - cosine_distance)
    
    Why cosine similarity:
    - Measures angle between vectors, not magnitude
    - Perfect for semantic similarity (direction matters, not length)
    - Normalized by default (range 0-1, where 1 = identical)
    - Invariant to document length differences
    - Standard metric for text embeddings
    
    Args:
        supabase: Supabase client
        cv_id: UUID of the CV to match (from cvs table)
        top_k: Number of top jobs to return
        threshold: Minimum similarity threshold (0.0 to 1.0)
    
    Returns:
        List of dictionaries with job_id, job_announcement_id, and similarity score
    """
    try:
        # Call the PostgreSQL function
        response = supabase.rpc(
            "match_jobs_by_cv_id",
            {
                "cv_id_param": cv_id,
                "match_count": top_k,
                "match_threshold": threshold
            }
        ).execute()
        
        return response.data
    
    except Exception as e:
        raise Exception(f"Failed to match jobs for CV {cv_id}: {str(e)}")


def get_top_jobs_for_cv_with_details(
    supabase: Client,
    cv_id: str,
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Find top-K most similar jobs with full job announcement details.
    
    Args:
        supabase: Supabase client
        cv_id: UUID of the CV to match
        top_k: Number of top jobs to return
        threshold: Minimum similarity threshold (0.0 to 1.0)
    
    Returns:
        List of dictionaries with job announcement details and similarity scores
    """
    # Get matching job IDs and scores
    matches = get_top_jobs_for_cv(supabase, cv_id, top_k, threshold)
    
    if not matches:
        return []
    
    # Fetch full job announcement details
    job_announcement_ids = [match["job_announcement_id"] for match in matches]
    response = supabase.table("job_announcements").select("*").in_(
        "id", job_announcement_ids
    ).execute()
    announcements_dict = {job["id"]: job for job in response.data}
    
    # Combine job details with similarity scores
    results = []
    for match in matches:
        announcement_id = match["job_announcement_id"]
        if announcement_id in announcements_dict:
            job_data = announcements_dict[announcement_id].copy()
            job_data["similarity_score"] = match["similarity"]
            job_data["job_id"] = match["job_id"]  # ID from jobs table
            results.append(job_data)
    
    return results


def get_top_cvs_for_job(
    supabase: Client,
    job_id: str,
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Find top-K most similar CVs for a given job (reverse matching).
    
    Args:
        supabase: Supabase client
        job_id: UUID of the job to match (from jobs table)
        top_k: Number of top CVs to return
        threshold: Minimum similarity threshold (0.0 to 1.0)
    
    Returns:
        List of dictionaries with cv_id, user_cv_id, and similarity score
    """
    try:
        response = supabase.rpc(
            "match_cvs_by_job_id",
            {
                "job_id_param": job_id,
                "match_count": top_k,
                "match_threshold": threshold
            }
        ).execute()
        
        return response.data
    
    except Exception as e:
        raise Exception(f"Failed to match CVs for job {job_id}: {str(e)}")


# ============================================================================
# Main execution functions
# ============================================================================

def run_full_job_pipeline():
    """Complete pipeline: populate jobs table and generate embeddings."""
    supabase, _ = init_clients()
    
    print("=== STEP 1: Populate jobs table ===")
    populate_stats = populate_jobs_from_announcements(supabase)
    print(f"Populate statistics: {populate_stats}\n")
    
    print("=== STEP 2: Generate embeddings ===")
    embed_stats = embed_all_jobs(supabase)
    print(f"Embedding statistics: {embed_stats}")


def run_full_cv_pipeline():
    """Complete pipeline: populate cvs table and generate embeddings."""
    supabase, _ = init_clients()
    
    print("=== STEP 1: Populate cvs table ===")
    populate_stats = populate_cvs_from_user_cvs(supabase)
    print(f"Populate statistics: {populate_stats}\n")
    
    print("=== STEP 2: Generate embeddings ===")
    embed_stats = embed_all_cvs(supabase)
    print(f"Embedding statistics: {embed_stats}")


def run_matching_example(cv_id: str, show_details: bool = False, threshold: float = 0.0, top_k: int = 5):
    """
    Example of matching a CV to top-K jobs.
    
    Args:
        cv_id: UUID of the CV (from cvs table)
        show_details: Whether to fetch and display full job details
        threshold: Minimum similarity threshold (0.0 to 1.0)
        top_k: Number of top jobs to return
    """
    supabase, _ = init_clients()
    
    print(f"\nFinding top-{top_k} jobs for CV ID: {cv_id}")
    print(f"Threshold: {threshold:.2f}")
    
    if show_details:
        matches = get_top_jobs_for_cv_with_details(supabase, cv_id, top_k=top_k, threshold=threshold)
        
        print("\nTop matching jobs with details:")
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. Similarity: {match['similarity_score']:.4f}")
            print(f"   Company: {match.get('entreprise_name', 'N/A')}")
            print(f"   Location: {match.get('work_location', 'N/A')}")
            print(f"   Contract: {match.get('contract_type', 'N/A')}")
            print(f"   Experience: {match.get('required_experience', 'N/A')}")
            print(f"   Description: {match.get('description', 'N/A')[:150]}...")
    else:
        matches = get_top_jobs_for_cv(supabase, cv_id, top_k=top_k, threshold=threshold)
        
        print("\nTop matching jobs:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. Job ID: {match['job_id']}, "
                  f"Announcement ID: {match['job_announcement_id']}, "
                  f"Similarity: {match['similarity']:.4f}")
    
    return matches


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cv_job_matcher.py populate_jobs    # Populate jobs table from announcements")
        print("  python cv_job_matcher.py embed_jobs       # Generate embeddings for jobs")
        print("  python cv_job_matcher.py full_jobs        # Run complete job pipeline")
        print()
        print("  python cv_job_matcher.py populate_cvs     # Populate cvs table from user_cvs")
        print("  python cv_job_matcher.py embed_cvs        # Generate embeddings for CVs")
        print("  python cv_job_matcher.py full_cvs         # Run complete CV pipeline")
        print()
        print("  python cv_job_matcher.py match <cv_id> [threshold] [top_k]")
        print("      # Match CV to jobs (threshold default: 0.0, top_k default: 5)")
        print("  python cv_job_matcher.py match_details <cv_id> [threshold] [top_k]")
        print("      # Match with full details")
        print()
        print("Examples:")
        print("  python cv_job_matcher.py match <cv_id>              # All matches, top 5")
        print("  python cv_job_matcher.py match <cv_id> 0.5          # Similarity >= 0.5, top 5")
        print("  python cv_job_matcher.py match <cv_id> 0.7 10       # Similarity >= 0.7, top 10")
        sys.exit(1)
    
    command = sys.argv[1]
    supabase, _ = init_clients()
    
    if command == "populate_jobs":
        stats = populate_jobs_from_announcements(supabase)
        print(f"\nFinal statistics: {stats}")
    
    elif command == "embed_jobs":
        stats = embed_all_jobs(supabase)
        print(f"\nFinal statistics: {stats}")
    
    elif command == "full_jobs":
        run_full_job_pipeline()
    
    elif command == "populate_cvs":
        stats = populate_cvs_from_user_cvs(supabase)
        print(f"\nFinal statistics: {stats}")
    
    elif command == "embed_cvs":
        stats = embed_all_cvs(supabase)
        print(f"\nFinal statistics: {stats}")
    
    elif command == "full_cvs":
        run_full_cv_pipeline()
    
    elif command == "match" and len(sys.argv) >= 3:
        cv_id = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        run_matching_example(cv_id, show_details=False, threshold=threshold, top_k=top_k)
    
    elif command == "match_details" and len(sys.argv) >= 3:
        cv_id = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        run_matching_example(cv_id, show_details=True, threshold=threshold, top_k=top_k)
    
    else:
        print("Invalid command or arguments")
        sys.exit(1)