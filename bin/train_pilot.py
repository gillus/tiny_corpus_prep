#!/usr/bin/env python3
"""
Pretrain a pilot model FROM SCRATCH on a corpus built by this repo.

Reads the output directory of build_corpus.py + tokenize_corpus.py
(dataset/ with packed input_ids, tokenizer/tokenizer.json) and trains a
Llama-architecture model with randomly initialized weights.

Defaults give ~50M parameters — the pipeline-shakedown pilot. Precision is
auto-selected: bf16 on Ampere+, fp16 with grad scaler on Turing (T4/RTX 20xx).

Requires: pip install torch transformers   (any recent versions)

Examples:
    # 50M pilot, one epoch over the corpus
    python bin/train_pilot.py --corpus-dir data/corpus_v1 --output-dir runs/pilot50m

    # Tiny dry run to validate the setup end-to-end
    python bin/train_pilot.py --corpus-dir data/corpus_v1 --output-dir runs/dry \\
        --hidden-size 64 --num-layers 2 --num-heads 2 --num-kv-heads 1 \\
        --intermediate-size 128 --max-steps 5

    # Resume after an interruption
    python bin/train_pilot.py --corpus-dir data/corpus_v1 --output-dir runs/pilot50m --resume
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# This script never needs TensorFlow, but a half-uninstalled TF (leftover
# tensorflow-* satellite packages creating an empty `tensorflow` namespace
# shim) crashes the transformers → accelerate → torch.utils.tensorboard
# import chain. Making TF unimportable is safe: tensorboard falls back to
# its bundled stub.
sys.modules["tensorflow"] = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description="Pretrain a pilot model from scratch")
    ap.add_argument("--corpus-dir", required=True,
                    help="build_corpus/tokenize_corpus output dir (contains dataset/, tokenizer/)")
    ap.add_argument("--output-dir", required=True, help="Where to write checkpoints and the final model")
    # Model (defaults ≈ 50M params with a 28k vocab)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=10)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--intermediate-size", type=int, default=1536)
    # Optimization
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.015)
    ap.add_argument("--micro-batch-size", type=int, default=16,
                    help="Sequences per device step (lower to 8-16 on 8GB cards)")
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="Gradient accumulation steps (global batch = micro * accum * seq_length tokens)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Optimizer steps; default = one epoch over the train split")
    ap.add_argument("--seed", type=int, default=42)
    # Bookkeeping
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in output-dir")
    ap.add_argument("--no-sample", action="store_true", help="Skip sample generations after training")
    return ap.parse_args()


def load_corpus_meta(corpus_dir: Path) -> dict:
    """seq_length / eos / pad from the tokenization manifest (with fallbacks)."""
    manifest_path = corpus_dir / "tokenization_manifest.json"
    meta = {"seq_length": None, "eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        meta["seq_length"] = manifest["stats"]["seq_length"]
        meta["eos_token"] = manifest["stats"].get("eos_token", meta["eos_token"])
    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    try:
        import torch
        from datasets import load_from_disk
        from transformers import (
            LlamaConfig,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        logger.error("Missing dependency: %s. Install with: pip install torch transformers datasets", e)
        return 1

    set_seed(args.seed)
    corpus_dir = Path(args.corpus_dir)
    tokenizer_file = corpus_dir / "tokenizer" / "tokenizer.json"
    dataset_dir = corpus_dir / "dataset"
    if not tokenizer_file.exists() or not dataset_dir.exists():
        logger.error(
            "Corpus dir %s must contain tokenizer/tokenizer.json and dataset/ — "
            "run build_corpus.py and tokenize_corpus.py first.", corpus_dir,
        )
        return 1

    meta = load_corpus_meta(corpus_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        eos_token=meta["eos_token"],
        pad_token=meta["pad_token"],
        # Llama-style models take no token_type_ids; without this the generic
        # fast tokenizer emits them and model.generate() rejects the input.
        model_input_names=["input_ids", "attention_mask"],
    )

    ds = load_from_disk(str(dataset_dir))
    train_ds, val_ds = ds["train"], ds.get("validation")
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None
    seq_length = meta["seq_length"] or len(train_ds[0]["input_ids"])
    logger.info(
        "Dataset: %d train / %s val sequences of %d tokens (%.1fM train tokens)",
        len(train_ds), len(val_ds) if val_ds is not None else 0,
        seq_length, len(train_ds) * seq_length / 1e6,
    )

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=seq_length,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)  # random init — training from scratch
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.1fM parameters (vocab %d)", n_params / 1e6, len(tokenizer))

    # Precision: bf16 on Ampere+, fp16 on Turing (T4 / RTX 20xx), fp32 on CPU.
    # Require NATIVE bf16 — recent torch reports emulated bf16 on Turing,
    # which runs without tensor cores and is much slower than fp16 there.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older torch without the parameter
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    else:
        use_bf16 = False
    use_fp16 = use_cuda and not use_bf16
    logger.info(
        "Device: %s | precision: %s",
        torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
    )

    tokens_per_step = args.micro_batch_size * args.grad_accum * seq_length
    if args.max_steps is None:
        max_steps = max(1, len(train_ds) // (args.micro_batch_size * args.grad_accum))
    else:
        max_steps = args.max_steps
    logger.info(
        "Training: %d steps x %.0fk tokens/step = %.2fB tokens (%.0f tokens/param)",
        max_steps, tokens_per_step / 1e3, max_steps * tokens_per_step / 1e9,
        max_steps * tokens_per_step / n_params,
    )

    def collate(features):
        ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    # Final save: model + tokenizer together, loadable with from_pretrained
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy(tokenizer_file, final_dir / "tokenizer.json")
    logger.info("Final model saved to %s", final_dir)

    if val_ds is not None:
        metrics = trainer.evaluate()
        logger.info("Final validation loss: %.4f", metrics["eval_loss"])

    if not args.no_sample:
        logger.info("--- sample generations ---")
        model.eval()
        device = model.device
        for prompt in ["The cat sat on the", '<household id="', "day: monday |"]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            room = seq_length - inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=min(80, room), do_sample=True,
                    temperature=0.8, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            logger.info("PROMPT %r →\n%s\n", prompt, text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
