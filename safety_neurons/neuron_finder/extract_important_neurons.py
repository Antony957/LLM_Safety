import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from constant import modelpath, tokenizerpath

from findneurons.lib.prune import (
    prune_wanda,
    prune_wanda_decouple_activations,
)

"""
model="llama2"
method="wanda"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --save $save_dir \
    --dump_wanda_score
"""

from findneurons.lib.model_wrapper import prune_wanda_v2, prune_wandg
from findneurons.lib.model_wrapper_low import make_low_rank
from findneurons.lib.eval import eval_ppl, eval_zero_shot, eval_attack

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"


def get_llm(model_name, cache_dir="llm_weights"):
    if model_name in [
        "llama2",
        "llama3",
        "vicuna",
        "guanaco",
    ]:
        model = AutoModelForCausalLM.from_pretrained(
            modelpath[model_name],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0.5, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type",
        type=str,
        choices=["unstructured", "4:8", "2:4"],
        default="unstructured",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "wanda",
            "wandg"
        ],
    )

    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "alpaca",
            "align",
        ],
        default="alpaca_cleaned_no_safety",
    )
    parser.add_argument("--use_diff", action="store_true")
    parser.add_argument("--neg_prune", action="store_true")
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument(
        "--prune_part",
        action="store_true",
        help="whether to only prune the layer with lower jaccard index",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_utility",
        action="store_true",
        help="whether to decouple the align and utility when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_misalign",
        action="store_true",
        help="whether to decouple the align and misalign when computing the wanda score",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top q scored elements in the second set (align))",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=10,
        help="Use combined with attention_head, the top k heads to prune",
    )

    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--save", type=str, default='./score', help="Path to save results.")
    parser.add_argument(
        "--dump_wanda_score", action="store_true", help="Whether to dump wanda scores."
    )

    args = parser.parse_args()


    if args.dump_wanda_score:
        assert args.prune_method in [
            "wanda",
            "wanda_v2",
            "wandg",
        ], "dump_wanda_score only works with wanda wanda_v2 wandg"

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizerpath[args.model], use_fast=False
    )

    model_base = None



    device = torch.device("cuda:0")
    print("Disentangle:", args.disentangle)

    
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "wandg":
            prune_wandg(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                prune_data=args.prune_data,
            )

    ################################################################
    print("*" * 30)
    if not args.recover_from_base and args.sparsity_ratio > 0:
        # check_sparsity_layerwise(model)
        sparsity_ratio = check_sparsity(model)
    else:
        sparsity_ratio = args.sparsity_ratio
    print(f"sparsity sanity check {sparsity_ratio:.6f}")
    print("*" * 30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")


if __name__ == "__main__":
    main()
