from huggingface_hub import login
from iad_utils import load_qwen_7b, get_gsm8k_questions
from distillation import self_distill, distillation_dataset
from amplification import amplify
import argparse
import os
os.environ["WANDB_PROJECT"] = "IAD"

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=5, help='number of iterations of amplification and distillation')
args = parser.parse_args()

# Read token from file and login
with open("token.txt", "r") as f:
    hf_token = f.read().strip()
login(hf_token)

def main():
    model, tokenizer = load_qwen_7b()
    
    dataset = get_gsm8k_questions(split=None)
    dataset['train'] = dataset['train'].select(range(8))
    dataset['test'] = dataset['test'].select(range(4))
    
    output_dir = f"./iad/R1-Qwen-7B-gsm8k-"

    for i in range(args.iters):
        model.train()
        print(f"Running self-distillation, iteration {i}")
        distil_dataset = distillation_dataset(model, tokenizer, dataset)
        self_distill(model=model, tokenizer=tokenizer, dataset=distil_dataset, output_dir=output_dir+f"distillation{i}")
        
        print(f"Running amplification, iteration {i+1}")
        amplify(model=model, tokenizer=tokenizer, dataset=dataset, output_dir=output_dir+f"amplification{i+1}")
        
if __name__ == "__main__":
    main()