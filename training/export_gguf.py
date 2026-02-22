"""
Export trained model to GGUF format for Ollama/llama.cpp deployment — Multi-Model.

Exports 3 quantizations (Q4_K_M, Q5_K_M, Q8_0), each into its own directory
for upload to separate HuggingFace repos. Logs file sizes and quantization
details to W&B.

For Unsloth models (Qwen3/DeepSeek): uses Unsloth's save_pretrained_gguf.
For PEFT models (MobileLLM): merges adapters and saves HF weights for
    manual llama.cpp conversion.

Usage:
    python export_gguf.py --model-key qwen3-4b --grpo-checkpoint ./checkpoints/grpo/qwen3-4b
    python export_gguf.py --model-key mobilellm-r1-950m --grpo-checkpoint ./checkpoints/grpo/mobilellm-r1-950m
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import wandb

from model_configs import get_config, resolve_model_key, ALL_MODEL_KEYS, QUANTIZATIONS


MODELFILE_TEMPLATE_QWEN = """FROM ./{gguf_filename}

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

MODELFILE_TEMPLATE_GENERIC = """FROM ./{gguf_filename}

SYSTEM \\"\\"\\"You are a financial sentiment analyst. Analyze the given financial text and provide:
1. Your reasoning in <reasoning> tags
2. Your sentiment classification (positive, negative, or neutral) in <answer> tags

Always use this exact format:
<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>[positive/negative/neutral]</answer>\\"\\"\\"

PARAMETER stop "</answer>"
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


def get_modelfile_template(model_family: str) -> str:
    """Get the appropriate Modelfile template for the model family."""
    if model_family in ("qwen3", "deepseek"):
        return MODELFILE_TEMPLATE_QWEN
    return MODELFILE_TEMPLATE_GENERIC


def export_unsloth(model, tokenizer, output_dir: Path, model_name: str, quant: str) -> dict:
    """Export a single quantization using Unsloth and return metadata."""
    quant_upper = quant.upper()
    gguf_filename = f"{model_name}.{quant_upper}.gguf"
    quant_dir = output_dir / quant_upper
    quant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting {quant_upper} via Unsloth...")
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

    file_size_mb = get_file_size_mb(str(final_path)) if final_path else 0

    return {
        "quantization": quant_upper,
        "filename": gguf_filename,
        "file_size_mb": round(file_size_mb, 1),
        "file_size_gb": round(file_size_mb / 1024, 2),
        "export_time_sec": round(export_time, 1),
        "path": str(final_path),
        "dir": str(quant_dir),
    }


def export_peft_to_hf(grpo_checkpoint: str, output_dir: Path, model_name: str) -> dict:
    """
    For non-Unsloth models: merge PEFT adapters and save HF weights.
    GGUF conversion can be done manually with llama.cpp's convert script.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"\n  Merging PEFT adapters for {model_name}...")
    start = time.time()

    merged_dir = output_dir / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    try:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            grpo_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            grpo_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(grpo_checkpoint, trust_remote_code=True)

    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    export_time = time.time() - start

    total_size = sum(
        os.path.getsize(str(f))
        for f in merged_dir.rglob("*")
        if f.is_file()
    )

    print(f"  Merged HF weights saved to {merged_dir}")
    print(f"  To convert to GGUF, run:")
    print(f"    python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outtype q4_k_m")
    print(f"    python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outtype q5_k_m")
    print(f"    python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outtype q8_0")

    return {
        "merged_dir": str(merged_dir),
        "total_size_mb": round(total_size / (1024 * 1024), 1),
        "export_time_sec": round(export_time, 1),
    }


def upload_quant_to_hf(quant_dir: str, gguf_filename: str, modelfile_path: str, hf_repo: str):
    """Upload a single quantization to its own HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)

    # Upload GGUF file
    gguf_path = os.path.join(quant_dir, gguf_filename)
    if os.path.exists(gguf_path):
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=gguf_filename,
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"    Uploaded {gguf_filename} -> {hf_repo}")

    # Upload Modelfile
    if os.path.exists(modelfile_path):
        api.upload_file(
            path_or_fileobj=modelfile_path,
            path_in_repo="Modelfile",
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"    Uploaded Modelfile -> {hf_repo}")


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF (multi-model, 3 quants)")
    parser.add_argument("--model-key", required=True,
                        help=f"Model key: {', '.join(ALL_MODEL_KEYS)}")
    parser.add_argument("--grpo-checkpoint", required=True,
                        help="Path to GRPO checkpoint")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ./export/<model-key>)")
    parser.add_argument("--model-name", default=None,
                        help="Model name for files (default: FinSent-CoT-<ShortName>)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace after export")
    args = parser.parse_args()

    model_key = resolve_model_key(args.model_key)
    config = get_config(model_key)

    output_dir = Path(args.output_dir or f"./export/{model_key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"FinSent-CoT-{config['short_name']}"
    hf_repos = config["hf_repos"]

    print("=" * 70)
    print(f"FinSent-CoT GGUF Export — {config['short_name']}")
    print("=" * 70)
    print(f"  Model key:      {model_key}")
    print(f"  Backend:        {'Unsloth' if config['use_unsloth'] else 'PEFT (HF merge only)'}")
    print(f"  Quantizations:  {', '.join(QUANTIZATIONS)}")
    print(f"  HF repos:")
    for quant, repo in hf_repos.items():
        print(f"    {quant}: {repo}")
    print()

    # ─── Initialize W&B ─────────────────────────────────────────────────────
    wandb.init(
        project="FinSent-CoT",
        name=f"export-{config['short_name']}",
        tags=["export", "gguf", "quantization", model_key],
        config={
            "phase": "gguf_export",
            "model_key": model_key,
            "grpo_checkpoint": args.grpo_checkpoint,
            "model_name": model_name,
            "use_unsloth": config["use_unsloth"],
            "quantizations": QUANTIZATIONS,
            "hf_repos": hf_repos,
        },
    )

    if config["use_unsloth"]:
        # ─── Unsloth export: 3 GGUF quantizations ───────────────────────────
        from unsloth import FastLanguageModel

        print(f"\nLoading from {args.grpo_checkpoint}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.grpo_checkpoint,
            max_seq_length=config["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )

        template = get_modelfile_template(config["model_family"])
        exports = []

        for quant in QUANTIZATIONS:
            quant_lower = quant.lower()
            info = export_unsloth(model, tokenizer, output_dir, model_name, quant_lower)
            exports.append(info)
            print(f"    -> {info['filename']}: {info['file_size_gb']} GB ({info['export_time_sec']}s)")

            # Write Modelfile for this quantization
            quant_dir = output_dir / quant
            modelfile_path = quant_dir / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(template.format(gguf_filename=info["filename"]))

        # Also save merged HF weights
        merged_dir = output_dir / "merged_hf"
        print(f"\nSaving merged HF weights to {merged_dir}...")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

        # Log to W&B
        comparison_table = wandb.Table(
            columns=["quantization", "file_size_gb", "file_size_mb", "export_time_sec"],
            data=[[e["quantization"], e["file_size_gb"], e["file_size_mb"], e["export_time_sec"]] for e in exports],
        )
        wandb.log({"quantization_comparison": comparison_table})

        summary = {"recommended_quantization": "Q5_K_M"}
        for e in exports:
            summary[f"{e['quantization'].lower()}_size_gb"] = e["file_size_gb"]
        wandb.summary.update(summary)

        # Upload to HuggingFace (each quant to its own repo)
        if args.upload:
            print(f"\nUploading to HuggingFace (3 separate repos)...")
            for e in exports:
                quant = e["quantization"]
                repo = hf_repos[quant]
                quant_dir = output_dir / quant
                modelfile_path = quant_dir / "Modelfile"
                upload_quant_to_hf(str(quant_dir), e["filename"], str(modelfile_path), repo)

            # Also upload merged HF weights to the Q5_K_M repo (recommended)
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=str(merged_dir),
                path_in_repo="merged_hf",
                repo_id=hf_repos["Q5_K_M"],
                repo_type="model",
            )
            print(f"    Uploaded merged HF weights -> {hf_repos['Q5_K_M']}/merged_hf")

        print(f"\n{'='*70}")
        print("EXPORT COMPLETE")
        print(f"{'='*70}")
        for e in exports:
            rec = " (recommended)" if e["quantization"] == "Q5_K_M" else ""
            print(f"  {e['quantization']}: {e['file_size_gb']} GB{rec}")
        print(f"  HF weights: {merged_dir}")
        print(f"\nTo create Ollama model (using Q5_K_M):")
        print(f"  cd {output_dir}/Q5_K_M && ollama create finsent-{model_key} -f Modelfile")

    else:
        # ─── PEFT export: merge adapters, save HF weights ───────────────────
        merge_info = export_peft_to_hf(args.grpo_checkpoint, output_dir, model_name)

        wandb.summary.update({
            "merged_size_mb": merge_info["total_size_mb"],
            "export_method": "peft_merge",
            "note": "Use llama.cpp convert_hf_to_gguf.py for GGUF quantization into 3 repos",
        })

        print(f"\n{'='*70}")
        print("EXPORT COMPLETE (HF weights only — GGUF requires manual conversion)")
        print(f"{'='*70}")
        print(f"  Merged weights: {merge_info['merged_dir']}")
        print(f"  Total size: {merge_info['total_size_mb']} MB")
        print(f"\n  Convert to GGUF with llama.cpp:")
        for quant in QUANTIZATIONS:
            print(f"    python convert_hf_to_gguf.py {merge_info['merged_dir']} --outtype {quant.lower()}")
        print(f"\n  Then upload each to:")
        for quant, repo in hf_repos.items():
            print(f"    {quant}: {repo}")

    wandb.finish()
    print(f"\nExport complete for {config['short_name']}!")


if __name__ == "__main__":
    main()
