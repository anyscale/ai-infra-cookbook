import os
import math
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from datetime import datetime, timezone
from typing import Dict, Any, List

# Configuration
num_samples_to_process = 100_000  # Start small for testing
num_gpus = 8  # Match the GPU allocation in job.yaml
quality_threshold = 0.5  # Filter samples with quality score above this threshold

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/mnt/shared_storage/fineweb_curated/{timestamp}"

# Ask-LLM prompt based on DCLM paper methodology
ASK_LLM_PROMPT = """Below is an extract from a web page. Evaluate whether the page contains high-quality content suitable for training a language model.

The ideal training data should:
- Be well-written and grammatically correct
- Contain educational or informative content
- Be coherent and have clear context
- Not be spam, advertisements, or low-quality content
- Not contain excessive repetition or boilerplate text

Text:
{text}

Question: Is this text suitable for training a language model?
Answer with only 'Yes' or 'No':"""


# vLLM Engine Configuration
processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
    ),
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
            VLLM_DISABLE_COMPILE_CACHE="1",
        ),
    ),
    batch_size=32,
    max_concurrent_batches=4,
    accelerator_type="A10G",
    concurrency=num_gpus,
)


def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the Ask-LLM prompt for quality scoring."""
    # Truncate text to avoid exceeding token limits (roughly 2000 chars ~ 500-700 tokens)
    text = row.get("text", "")[:2000]

    return dict(
        messages=[
            {
                "role": "user",
                "content": ASK_LLM_PROMPT.format(text=text),
            }
        ],
        sampling_params=dict(
            temperature=0.0,  # Deterministic for consistent scoring
            max_tokens=5,  # Only need "Yes" or "No"
            logprobs=10,  # Request logprobs to extract probability
        ),
    )


def compute_yes_probability(logprobs: List) -> float:
    """
    Extract the probability of 'Yes' from logprobs.

    The Ask-LLM approach uses P(Yes) as the quality score.
    We look at the first token's logprobs and find the probability
    assigned to 'Yes' (or related tokens like 'yes', 'YES').
    """
    if not logprobs or len(logprobs) == 0:
        return 0.0

    # Get the first token's logprobs (the answer token)
    first_token_logprobs = logprobs[0]

    if not first_token_logprobs:
        return 0.0

    # Look for "Yes" variants in the logprobs
    yes_variants = {"Yes", "yes", "YES", " Yes", " yes"}
    no_variants = {"No", "no", "NO", " No", " no"}

    yes_logprob = None
    no_logprob = None

    # first_token_logprobs is typically a dict mapping token -> logprob
    # or a list of (token, logprob) tuples depending on vLLM version
    if isinstance(first_token_logprobs, dict):
        for token, logprob_info in first_token_logprobs.items():
            token_str = token if isinstance(token, str) else str(token)
            logprob_val = logprob_info if isinstance(logprob_info, (int, float)) else getattr(logprob_info, 'logprob', logprob_info)

            if token_str in yes_variants:
                yes_logprob = logprob_val
            elif token_str in no_variants:
                no_logprob = logprob_val
    elif isinstance(first_token_logprobs, list):
        for item in first_token_logprobs:
            if hasattr(item, 'decoded_token') and hasattr(item, 'logprob'):
                token_str = item.decoded_token
                logprob_val = item.logprob
            elif isinstance(item, dict):
                token_str = item.get('token', item.get('decoded_token', ''))
                logprob_val = item.get('logprob', 0)
            else:
                continue

            if token_str in yes_variants:
                yes_logprob = logprob_val
            elif token_str in no_variants:
                no_logprob = logprob_val

    # If we found both Yes and No, compute softmax probability
    if yes_logprob is not None and no_logprob is not None:
        # Softmax: P(Yes) = exp(yes_logprob) / (exp(yes_logprob) + exp(no_logprob))
        max_logprob = max(yes_logprob, no_logprob)
        yes_exp = math.exp(yes_logprob - max_logprob)
        no_exp = math.exp(no_logprob - max_logprob)
        return yes_exp / (yes_exp + no_exp)

    # If only Yes found, return its probability
    if yes_logprob is not None:
        return math.exp(yes_logprob)

    # Fallback: check the generated text
    return 0.0


def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quality score from LLM response."""
    # Get logprobs from the response
    logprobs = row.get("generated_logprobs", [])

    # Compute quality score
    quality_score = compute_yes_probability(logprobs)

    # Also check the generated text as a fallback
    generated_text = row.get("generated_text", "").strip().lower()
    if quality_score == 0.0:
        # Fallback: binary score based on generated text
        if generated_text.startswith("yes"):
            quality_score = 1.0
        elif generated_text.startswith("no"):
            quality_score = 0.0

    row["quality_score"] = quality_score

    # Clean up intermediate fields to save storage
    row.pop("generated_logprobs", None)
    row.pop("generated_text", None)
    row.pop("messages", None)
    row.pop("sampling_params", None)

    return row


def main():
    # Build the LLM processor
    llm_processor = build_llm_processor(
        processor_config,
        preprocess=preprocess,
        postprocess=postprocess,
    )

    # Load FineWeb-edu dataset from HuggingFace
    print(f"Loading FineWeb-edu dataset (limiting to {num_samples_to_process:,} samples)...")
    dataset = (
        ray.data.read_parquet(
            "hf://datasets/HuggingFaceFW/fineweb-edu/data/",
            file_extensions=["parquet"],
            filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
            concurrency=20,
        )
        .limit(num_samples_to_process)
        .repartition(target_num_rows_per_block=500)
    )

    print("Applying Ask-LLM quality scoring with Qwen2.5-3B-Instruct...")
    # Apply LLM-based quality scoring
    dataset = llm_processor(dataset)

    # Filter by quality threshold
    print(f"Filtering samples with quality_score > {quality_threshold}...")
    dataset = dataset.filter(lambda row: row.get("quality_score", 0) > quality_threshold)

    # Write curated dataset to parquet
    print(f"Writing curated dataset to {output_path}...")
    dataset.write_parquet(output_path)

    print(f"Data curation complete. Output written to {output_path}")
    print(f"Final sample count: {dataset.count()}")


if __name__ == "__main__":
    main()
