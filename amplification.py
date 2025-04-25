from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset, Dataset
from huggingface_hub import login
from iad_utils import load_qwen_7b, get_gsm8k_questions, extract_xml_answer, GSM8K_PROMPT
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import os
os.environ["WANDB_PROJECT"] = "IAD"

XML_COT_FORMAT = """
... 
</think>-
...
<answer>
{answer}
</answer>
"""


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    print('-'*20, f"\nRESPONSE:\n{completions[0]}", f"\nEXTRACTED:{extracted_responses[0]}\nANSWER:{answer[0]}")
    # print('-'*20, f"PROMPT:\n{prompts[0]}\nRESPONSE:\n{completions[0]}", f"\nEXTRACTED:{extracted_responses[0]}\nANSWER:{answer[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^.*</think>\n.*\n<answer>\n.*\n</answer>\n"
    matches = [re.match(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^.*</think>.*<answer>.*</answer>.*"
    matches = [re.match(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("\n</think>\n") == 1:
        count += 0.166
    if text.count("\n<answer>\n") == 1:
        count += 0.166
        # count -= len(text.split("\n</answer>\n")[-1])*0.001
        count -= len(text.split("\n<answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.166
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]

max_prompt_length = 512

def amplify(model, tokenizer, dataset, output_dir):
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = 1024,
        num_train_epochs = 1,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "R1-Qwen-7B-gsm8k-amplification1",
        run_name=output_dir
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset['train'],
    )
    trainer.train()

    trainer.save_model(output_dir)
    
    return model, tokenizer

if __name__ == "__main__":

    # Read token from file and login
    with open("token.txt", "r") as f:
        hf_token = f.read().strip()
    login(hf_token)

    dataset = get_gsm8k_questions()
    
    model, tokenizer = load_qwen_7b(load_adapter="./R1-Qwen-7B-gsm8k-distillation0")
    amplify(model, tokenizer, dataset, output_dir="R1-Qwen-7B-gsm8k-amplification1")

    """<a name="Inference"></a>
    ### Inference
    Now let's try the model we just trained! First, let's first try the model without any GRPO trained:
    """

    text = GSM8K_PROMPT.format(question="Calculate pi.")

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text

    print(output)

    """And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

    model.save_lora("grpo_saved_lora")

    """Now we load the LoRA and test:"""

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        # sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text

    print(output)