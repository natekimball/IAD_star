from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
# For supervised fine-tuning with Unsloth
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import copy
from iad_utils import get_gsm8k_questions, GSM8K_PROMPT
import os
os.environ["WANDB_PROJECT"] = "IAD"

# Configuration
# MODEL_NAME = "unsloth/Llama-3.1-8B"
# MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
response_template = " ### Answer:"

# TODO: when loading a model run inference_model = ... after loading peft

def distillation_dataset(model, tokenizer, dataset):
    inference_model = FastLanguageModel.for_inference(copy.deepcopy(model))
    
    def batched_preprocess_fn(examples):
        # questions = examples['question']
        # prompts = [GSM8K_PROMPT.format(question=question) for question in questions]
        prompts = examples['prompt']
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to(inference_model.device)
        outputs = inference_model.generate(inputs)
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        texts = []
        for prompt, response in zip(prompts, responses):
            answer = response.split('</think>')[-1]
            # answer = response[response.rfind('</think>'):]
            # full_text = f"### Question: {question}\n ### Reply: {answer}"
            full_text = f"{prompt}\n{response_template} {answer}"
            texts.append(full_text)
        return {'text': texts}
    
    # new_dataset = DatasetDict()
    # new_dataset['train'] = dataset['train'].map(batched_preprocess_fn, remove_columns=dataset['train'].column_names, batched=True, batch_size=8)
    # new_dataset['test'] = dataset['train'].map(lambda x: {'text': f"{x['prompt']}\n ### Answer: {x['answer']}"})
    # return new_dataset
    
    return dataset.map(batched_preprocess_fn, remove_columns=dataset['train'].column_names, batched=True, batch_size=8)



def self_distill(model, tokenizer, dataset, output_dir="output"):
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        # run_name=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
        dataset_text_field="text",  # Field containing the text to train on
        max_seq_length=2048,
        args=training_args,
        data_collator=collator,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    
    return model, tokenizer

def main():
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
        gpu_memory_utilization = 0.4, # Reduce if out of memory
    )

    # inference_model = FastLanguageModel.for_inference(copy.deepcopy(model))

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
    # dataset = load_dataset("openai/gsm8k", "main")
    dataset = get_gsm8k_questions()
    dataset['train'] = dataset['train'].select(range(100))
    dataset['test'] = dataset['test'].select(range(20))

    # dataset = dataset.map(batched_preprocess_fn, remove_columns=dataset['train'].column_names, batched=True, batch_size=8)
    dataset = distillation_dataset(model, tokenizer, dataset)
    self_distill(model=model, tokenizer=tokenizer, dataset=dataset, output_dir="./R1-Qwen-7B-gsm8k-distillation0")

if __name__ == "__main__":
    main()
