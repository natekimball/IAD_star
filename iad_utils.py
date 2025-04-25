from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import torch

# GSM8K_PROMPT = """
# ### Instruction: 
# You are a helpful mathematical problem solver designed to break down and solve grade school math problems. Think step by step, then solve the given problem and give your final numerical answer surrounded by answer tags, e.g.:
# thought process...
# </think>
# If Amanda has 203 toys and gives away 2 of them, Amanda now has 203 - 2 = 110 toys.
# <answer>
# 110
# </answer>

# #### Problem:
# {question}
# """
GSM8K_PROMPT = """
### Instruction: 
You are a helpful mathematical problem solver designed to break down and solve grade school math problems. Think step by step, then solve the given problem and give your final numerical answer surrounded by answer tags, e.g.:
<answer>
110
</answer>

#### Problem:
{question}
"""


def load_qwen_7b(load_adapter=None):
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ] # Remove QKVO if out of memory

    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        dtype=torch.float16,
        # fast_inference = True, # Enable vLLM fast inference
        # max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
        # supported_lora_modules = target_modules
    )
    # TODO: load saved model from distillation

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = target_modules,
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    
    if load_adapter is not None:
        print('Loading LoRA adapter from:', load_adapter)
        model.load_adapter(load_adapter, "adapter_model")
    
    return model, tokenizer


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = None) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main', split=split) # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': GSM8K_PROMPT.format(question=x['question']), # TODO: do I need to add an instruction?
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore