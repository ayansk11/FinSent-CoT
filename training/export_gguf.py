"""
Export trained model to GGUF format for Ollama/llama.cpp deployment.

Exports BOTH Q5_K_M and Q4_K_M quantizations so they can be benchmarked
side-by-side. Logs file sizes and quantization details to W&B.

Usage:
    python export_gguf.py --grpo-checkpoint ./checkpoints/grpo --output-dir ./export
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import wandb
from unsloth import FastLanguageModel


MODELFILE_TEMPLATE = """FROM ./{gguf_filename}

# Qwen3 chat template — prefill past <think> to skip empty thinking block
TEMPLATE \\"\\"\\"<|im_start|>system
You are a financial sentiment analyst. Analyze the given financial text and provide:
1. Your reasoning in <reasoning> tags
2. Your sentiment classification (positive, negative, or neutral) in <answer> tags

Always use this exact format:
<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>[positive/negative/neutral]</answer><|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>
</think>

\\"\\"\\"

# Critical stop tokens
PARAMETER stop "<|im_end|>"
PARAMETER stop "</answer>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_start|>"

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1024
PARAMETER num_predict 512
"""


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def export_quantization(model, tokenizer, output_dir: Path, model_name: str, quant: str) -> dict:
    """Export a single quantization and return metadata."""
    gguf_filename = f"{model_name}.{quant.upper()}.gguf"
    quant_dir = output_dir / quant.upper()
    quant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting {quant.upper()}...")
    start = time.time()

    model.save_pretrained_gguf(
        str(quant_dir),
        tokenizer,
        quantization_method=quant,
    )

    export_time = time.time() - start

    # Find and rename the GGUF file
    final_path = None
    for f in quant_dir.glob("*.gguf"):
        if f.name != gguf_filename:
            dest = quant_dir / gguf_filename
            shutil.move(str(f), str(dest))
            final_path = dest
        else:
            final_path = f

    # Also copy to top-level output dir
    if final_path:
        top_level = output_dir / gguf_filename
        shutil.copy2(str(final_path), str(top_level))
        final_path = top_level

    # Write Modelfile for this quantization
    modelfile_path = quant_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(MODELFILE_TEMPLATE.format(gguf_filename=gguf_filename))

    file_size_mb = get_file_size_mb(str(final_path)) if final_path else 0

    return {
        "quantization": quant.upper(),
        "filename": gguf_filename,
        "file_size_mb": round(file_size_mb, 1),
        "file_size_gb": round(file_size_mb / 1024, 2),
        "export_time_sec": round(export_time, 1),
        "path": str(final_path),
        "modelfile_path": str(modelfile_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument("--grpo-checkpoint", required=True,
                        help="Path to GRPO checkpoint")
    parser.add_argument("--output-dir", default="./export",
                        help="Output directory")
    parser.add_argument("--model-name", default="FinSent-CoT-Qwen3-4B",
                        help="Model name for files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FinSent-CoT GGUF Export (Q5_K_M + Q4_K_M)")
    print("=" * 70)

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name="export-gguf-dual-quant",
        tags=["export", "gguf", "quantization"],
        config={
            "phase": "gguf_export",
            "grpo_checkpoint": args.grpo_checkpoint,
            "model_name": args.model_name,
            "quantizations": ["q5_k_m", "q4_k_m"],
        },
    )

    # Load trained model
    print(f"\nLoading from {args.grpo_checkpoint}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.grpo_checkpoint,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Export both quantizations
    exports = []

    # Q5_K_M — higher quality, recommended for M4
    q5_info = export_quantization(model, tokenizer, output_dir, args.model_name, "q5_k_m")
    exports.append(q5_info)
    print(f"    -> {q5_info['filename']}: {q5_info['file_size_gb']} GB ({q5_info['export_time_sec']}s)")

    # Q4_K_M — smaller, slightly lower quality
    q4_info = export_quantization(model, tokenizer, output_dir, args.model_name, "q4_k_m")
    exports.append(q4_info)
    print(f"    -> {q4_info['filename']}: {q4_info['file_size_gb']} GB ({q4_info['export_time_sec']}s)")

    # Also save merged HF weights for HuggingFace upload
    merged_dir = output_dir / "merged_hf"
    print(f"\nSaving merged HF weights to {merged_dir}...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # ─── Log comparison to W&B ───────────────────────────────────────────────
    comparison_table = wandb.Table(
        columns=["quantization", "file_size_gb", "file_size_mb", "export_time_sec",
                 "quality_note", "recommended_for"],
        data=[
            [
                q5_info["quantization"],
                q5_info["file_size_gb"],
                q5_info["file_size_mb"],
                q5_info["export_time_sec"],
                "~0.5-1% perplexity increase vs FP16",
                "Mac M4 (16GB) — best quality/size balance",
            ],
            [
                q4_info["quantization"],
                q4_info["file_size_gb"],
                q4_info["file_size_mb"],
                q4_info["export_time_sec"],
                "~1.5-2% perplexity increase vs FP16",
                "Memory-constrained devices (8GB RAM)",
            ],
        ],
    )
    wandb.log({"quantization_comparison": comparison_table})

    wandb.summary.update({
        "q5_k_m_size_gb": q5_info["file_size_gb"],
        "q4_k_m_size_gb": q4_info["file_size_gb"],
        "size_difference_mb": q5_info["file_size_mb"] - q4_info["file_size_mb"],
        "recommended_quantization": "Q5_K_M",
        "recommendation_reason": (
            "Financial sentiment requires precise reasoning. "
            "Q5_K_M has ~0.5-1% perplexity increase vs FP16, while Q4_K_M has ~1.5-2%. "
            "The ~400MB size difference is negligible on M4 Air with 16GB RAM."
        ),
    })

    wandb.finish()

    # Write the default Modelfile pointing to Q5_K_M (recommended)
    default_modelfile = output_dir / "Modelfile"
    with open(default_modelfile, "w") as f:
        f.write(MODELFILE_TEMPLATE.format(gguf_filename=q5_info["filename"]))

    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  Q5_K_M (recommended): {q5_info['file_size_gb']} GB")
    print(f"  Q4_K_M (smaller):     {q4_info['file_size_gb']} GB")
    print(f"  HF weights:           {merged_dir}")
    print(f"  Default Modelfile:    {default_modelfile} (uses Q5_K_M)")
    print(f"\nTo create Ollama model:")
    print(f"  cd {output_dir} && ollama create finsent-cot -f Modelfile")


if __name__ == "__main__":
    main()
