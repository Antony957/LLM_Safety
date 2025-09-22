import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
from SafeDecoding.utils.safe_decoding import SafeDecoding
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from SafeDecoding.utils.ppl_calculator import PPL_Calculator

import json
from model.EEG import EEG
import torch
from model.utils import *
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Defense manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="vicuna")
    parser.add_argument("--attacker", type=str, default="GCG")
    parser.add_argument("--defender", type=str, default='EEG')

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", type=bool, default=True)

    parser.add_argument("--alpha", type = float, default=0.75)
    parser.add_argument("--t", type = int, default=12)

    return parser.parse_args()

args = get_args()


# Load model and template
if args.model_name == "vicuna":
    model_name = "lmsys/vicuna-7b-v1.5"
    prototype_name = 'model/vicuna_prototype.pt'
elif args.model_name == "llama2":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prototype_name = 'model/llama_prototype.pt'
elif args.model_name == "guanaco":
    model_name = "TheBloke/guanaco-7B-HF"
    prototype_name = 'model/guanaco_prototype.pt'
else:
    raise ValueError("Unsupported model name.")

def load_model(self, model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

ego = EGO(model_name, prototype_name, {'max_new_tokens': args.max_new_tokens, 'do_sample': args.do_sample})
if args.attacker == 'GCG' and args.model_name == 'llama2':
    ego.set_gcg_system_prompt()

# Initialize defenders
# Load PPL Calculator

elif args.defender not in ['EGO', 'NoDefense']:
    raise ValueError("Unsupported defender name.")


attack_prompts = load_dataset('antony957/EGO_Attacker')
attack_prompts = attack_prompts[args.attacker]


output_list = []

# Start generation
for prompt in tqdm(attack_prompts):
    user_prompt = prompt


    if args.defender == 'EEG':
        output = ego.safe_generate(args.t, args.alpha, user_prompt)
    else:
        output = ego.generate(user_prompt)
    is_successful = check_jailbreak_success(output)
    output_list.append({'prompt': prompt, 'output': output, 'is_successful':is_successful})

with open('exp_output.json', 'w') as f:
    json.dump(output_list, f, indent=4)

