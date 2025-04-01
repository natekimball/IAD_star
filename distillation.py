import torch
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
# For supervised fine-tuning with Unsloth
from datasets import load_dataset
from huggingface_hub import login
import copy
from iad_utils import GSM8K_PROMPT

# Configuration
# MODEL_NAME = "unsloth/Llama-3.1-8B"
# MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

# Read token from file and login
with open("token.txt", "r") as f:
    hf_token = f.read().strip()
login(hf_token)

# Load model and tokenizer with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

inference_model = FastLanguageModel.for_inference(copy.deepcopy(model))

# Add LoRA adapters for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (adjustable: 8, 16, 32, etc.)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized for Unsloth
    bias="none",
    use_gradient_checkpointing=True,  # Memory efficiency
)

# Load GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main")

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

def preprocess_fn(example):
    question = example['question']
    inputs = tokenizer(question, truncation=True, max_length=1024, return_tensors='pt').input_ids.to(inference_model.device)
    outputs = inference_model.generate(inputs)
    response = tokenizer.decode(outputs[0])
    answer = response.split('</think>')[-1]
    full_text = f"### Question: {question}\n ### Answer: {answer}"
    return {'text': full_text}
    # return full_text


def batched_preprocess_fn(examples):
    questions = examples['question']
    prompts = [GSM8K_PROMPT.format(question=question) for question in questions]
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to(inference_model.device)
    outputs = inference_model.generate(inputs)
    responses = tokenizer.batch_decode(outputs)
    texts = []
    for prompt, response in zip(prompts, responses):
        answer = response.split('</think>')[-1]
        # full_text = f"### Question: {question}\n ### Answer: {answer}"
        full_text = prompt + answer
        texts.append(full_text)
    return {'text': texts}

dataset['train'] = dataset['train'].select(range(512))
dataset['test'] = dataset['test'].select(range(128))

dataset = dataset.map(batched_preprocess_fn, remove_columns=dataset['train'].column_names, batched=True, batch_size=32)

def run_training(output_dir="output"):
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,  # Use mixed precision
    )

    # Initialize SFTTrainer (Unsloth-compatible trainer)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        max_seq_length=2048,
        args=training_args,
        # formatting_func=preprocess_fn,
        data_collator=collator,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(output_dir)

def main():
    run_training(output_dir="./R1-Qwen-7B-gsm8k-distillation0")

if __name__ == "__main__":
    main()
