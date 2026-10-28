"""SFT with LoRA on the OLMo base model: teach it to answer like me.

    uv run python train_sft.py --variant A          # my data only
    uv run python train_sft.py --variant B          # mine x3 + classmates x1, speaker-tagged
    uv run python train_sft.py --variant A --limit 500                       # debug run
    uv run python train_sft.py --variant A --max-steps 20 --data <dir>       # smoke test

The data is TRL's prompt-completion format, so the loss is only on the answer: the model
learns to produce my words, while the professor's ask, classmates' replies and the LOs/HCs are
context it reads but is never trained to say. Writes the adapter to checkpoints/sft-{variant}/
(gitignored: an adapter can regurgitate its training text).
"""

import argparse
import time
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

import common


def dataset(variant, split, data):
    rows = common.sft_rows(variant, split, data)
    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--epochs", type=float, default=1.0, help="my examples already repeat 3x")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, help="train on the first N shuffled rows only")
    ap.add_argument("--batch", type=int, default=4, help="per forward pass")
    ap.add_argument("--effective-batch", type=int, default=common.EFFECTIVE_BATCH,
                    help="examples per optimizer step; gradient accumulation makes up the rest")
    ap.add_argument("--attn", default="sdpa", help="sdpa (any device) or flash_attention_2 (CUDA)")
    ap.add_argument("--packing", action="store_true",
                    help="pack examples without padding; only with FlashAttention, which keeps "
                         "packed examples from seeing each other")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="faster when memory allows (a big GPU); results are the same")
    args = ap.parse_args()
    if args.packing and "flash" not in args.attn:
        ap.error("--packing needs a FlashAttention --attn: with sdpa, packed examples would attend to each other")
    out = args.out or common.CHECKPOINTS / f"sft-{args.variant}"

    train = dataset(args.variant, "train", args.data).shuffle(seed=0)
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    val = dataset(args.variant, "val", args.data)
    print(f"variant {args.variant}: {len(train)} train rows, {len(val)} val rows")

    config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=max(1, args.effective_batch // args.batch),
        learning_rate=common.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=common.MAX_LENGTH,
        gradient_checkpointing=not args.no_checkpointing,
        packing=args.packing,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        include_num_input_tokens_seen="non_padding",
        report_to="none",  # no wandb: logged samples would carry class text
        seed=0,
    )
    trainer = SFTTrainer(model=common.load_model(attn=args.attn), args=config, train_dataset=train,
                         eval_dataset=val, processing_class=common.load_tokenizer(),
                         peft_config=common.lora_config())
    trainer.model.print_trainable_parameters()
    start = time.time()
    trainer.train()
    seconds = time.time() - start
    tokens = trainer.state.num_input_tokens_seen or 0
    print(f"trained {trainer.state.global_step} steps in {seconds / 60:.1f} min"
          + (f", {tokens / seconds:.0f} tokens/s" if tokens else ""))
    print(f"val loss: {trainer.evaluate()['eval_loss']:.3f}")
    trainer.save_model(str(out))
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
