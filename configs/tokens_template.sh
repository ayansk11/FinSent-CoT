#!/bin/bash
# ─── Auth tokens for Big Red 200 ────────────────────────────────────────────
# Copy this to /N/scratch/ayshaikh/.tokens and fill in your keys.
# This file is sourced by all SLURM scripts.
#
# IMPORTANT: Do NOT commit this file to git. It's in .gitignore.
# ─────────────────────────────────────────────────────────────────────────────

# HuggingFace — get from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Weights & Biases — get from https://wandb.ai/settings (API Keys section)
# IMPORTANT: Regenerate your key if it was ever exposed in chat/logs
export WANDB_API_KEY="your_wandb_api_key_here"
