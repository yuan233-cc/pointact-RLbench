"""Isolated text-only Qwen benchmark; does not run or modify a trainer."""

import argparse
import time

import torch
from transformers import AutoTokenizer

from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def timed_forward(model, input_ids, attention_mask):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unique-prompts", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.unique_prompts <= args.batch_size:
        parser.error("expected 1 <= unique-prompts <= batch-size")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete RLBench task number {i}: move the object."}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for i in range(args.unique_prompts)
    ]
    samples = [prompts[i % len(prompts)] for i in range(args.batch_size)]
    all_tokens = tokenizer(samples, padding=True, return_tensors="pt")
    unique_tokens = tokenizer(prompts, padding=True, return_tensors="pt")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.train()  # Match the trainer's mode even though its Qwen weights are frozen.
    model.requires_grad_(False)
    print(f"attention_dropout={model.model.language_model.config.attention_dropout}")
    print(f"batch={args.batch_size} unique={args.unique_prompts} seq={all_tokens['input_ids'].shape[1]}")

    for _ in range(2):
        timed_forward(model, **all_tokens)
        timed_forward(model, **unique_tokens)

    full_times = []
    unique_times = []
    for _ in range(args.repeats):
        full_output, full_seconds = timed_forward(model, **all_tokens)
        unique_output, unique_seconds = timed_forward(model, **unique_tokens)
        full_times.append(full_seconds)
        unique_times.append(unique_seconds)

    # Equal-length prompts make padded hidden-state comparison unambiguous.
    indexed = unique_output[torch.arange(args.batch_size, device="cuda") % args.unique_prompts]
    valid = all_tokens["attention_mask"].to("cuda").bool().unsqueeze(-1)
    error = (full_output - indexed).abs().masked_select(valid)
    full_again, _ = timed_forward(model, **all_tokens)
    repeat_error = (full_output - full_again).abs().masked_select(valid)
    print(f"same_batch_repeat_max_abs_error={repeat_error.max().item():.8f}")
    print(f"max_abs_error={error.max().item():.8f} mean_abs_error={error.mean().item():.8f}")
    same_shape_cache = {}
    for prompt_idx, prompt in enumerate(prompts):
        repeated = tokenizer([prompt] * args.batch_size, padding=True, return_tensors="pt")
        if repeated["input_ids"].shape[1] != all_tokens["input_ids"].shape[1]:
            print("same_shape_cache_skipped=variable_prompt_length")
            break
        repeated_output, _ = timed_forward(model, **repeated)
        same_shape_cache[prompt_idx] = repeated_output[0]
    if len(same_shape_cache) == len(prompts):
        same_shape_indexed = torch.stack(
            [same_shape_cache[i % args.unique_prompts] for i in range(args.batch_size)]
        )
        same_shape_error = (full_output - same_shape_indexed).abs().masked_select(valid)
        print(f"same_shape_cache_max_abs_error={same_shape_error.max().item():.8f}")
    full_ms = 1000 * sum(full_times) / len(full_times)
    unique_ms = 1000 * sum(unique_times) / len(unique_times)
    print(f"full_forward_ms={full_ms:.2f} unique_forward_ms={unique_ms:.2f} speedup={full_ms / unique_ms:.2f}x")


if __name__ == "__main__":
    main()
