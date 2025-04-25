import unsloth
import os
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import time

# Check for CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# The correct get_prompt function for spam detection
def get_prompt(video, author, date, text, clas):
    return (f"You are an AI agend made to identify comments made under a youtube video. Your task is to return \"Yes\" if you consider that the linked comment should be deleted and \"No\" otherwize.\nYour output must be strictly be \"Yes\" or \"No\".\n\nTo do so, you have acces to the author name, the date the comment was posted and the content of the message.\nShould be deleted any filter evasion, hate speach, advertisement to something, and so on.\nremember that a negative comment is not necessary spam and that trolling comments should not be consider as spam. \n\nAuthor name: {author}\n\nComment date: {date}\n\nBEGIN COMMENT CONTENT\n{text}\nEND COMENT CONTENT", "Yes" if clas == 1 else "No")

# Load dataset from CSV and create instruction format
def load_dataset(file_path="video_dataset.csv"):
    df = pd.read_csv(file_path, delimiter=",")
    
    # Create instruction-response format
    def format_data(row):
        # Extract data from the DataFrame row
        video = row["VIDEO"]
        author = row["AUTHOR"]
        date = row["DATE"]
        text = row["TEXT"]
        clas = row["CLASS"]  # Assuming CLASS is 1 for spam, 0 for not spam
        
        # Get prompt and response using the correct function
        prompt, response = get_prompt(video, author, date, text, clas)
        
        return {
            "prompt": prompt,
            "response": response
        }
    
    # Apply formatting to each row
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append(format_data(row))
    
    # Create dataset from formatted data
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in formatted_data],
        "response": [item["response"] for item in formatted_data]
    })
    
    return dataset

# Set up model parameters
max_seq_length = 512  # Reduced for 4060 memory constraints
lora_rank = 128 # Adjusted for performance/memory tradeoff
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Changed to 3B instruct model

# Load the model with 4-bit quantization optimized for your 4060
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # Use 4-bit quantization for your 4060
    fast_inference=False,  # Disable vLLM as it requires more memory
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.95,  # Adjust based on your 4060's VRAM
)

# Set up LoRA for efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank*2,
    use_gradient_checkpointing="unsloth",  # Enable efficient training
    random_state=42,
)

# Load and prepare the dataset
dataset = load_dataset()
print(f"Dataset size: {len(dataset)} samples")

# Function to format the data for instruction tuning
def formatting_func(example):
    # Create instruction format for the instruct model
    return f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"

# Set up training arguments with conservative parameters for your 4060
training_args = TrainingArguments(
    output_dir="youtube_spam_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=6e-5,
    weight_decay=1.0,
    fp16=False,  # Disable mixed precision since we're using 4-bit
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    report_to="none",  # Disable W&B reporting
)

# Configure the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="prompt",  # Use prompt field for input
    args=training_args,
    packing=True,  # Enable packing for more efficient training
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
)

# Start training
print("Starting training...")
trainer.train()

# Save the trained model
output_dir = "modelm0.4.0"
os.makedirs(output_dir, exist_ok=True)

# Code for inference with the trained model
def detect_spam(video, author, date, comment_text):
    """
    Use the trained model to detect if a comment is spam
    """
    # Use the same get_prompt function as during training
    prompt, _ = get_prompt(video, author, date, comment_text, 0)  # Class doesn't matter for inference prompt
    
    # Format with the instruct model's expected tokens
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate with LoRA
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # We only need a short response (Yes/No)
            temperature=0.7,    # Lower temperature for more deterministic output
            top_p=0.95,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    response_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    response_end = generated_text.find("<|im_end|>", response_start)
    if response_end == -1:
        response = generated_text[response_start:]
    else:
        response = generated_text[response_start:response_end]
    
    # Clean up the response to get just Yes/No
    response = response.strip()
    
    return response

# Test the model with a sample prompt
if __name__ == "__main__":
    print("\nTesting the trained model:")
    test_video = "How to make chocolate chip cookies"
    test_author = "CookieFan123"
    test_date = "2023-05-15"
    test_comment = "Check out my awesome new website with FREE iPhone giveaways! Click now: bit.ly/notascam"
    
    result = detect_spam(test_video, test_author, test_date, test_comment)
    print(f"Is this comment spam? {result}")

    # Test with a non-spam comment
    test_comment2 = "Great recipe! I tried it and my family loved the cookies. Thanks for sharing!"
    result2 = detect_spam(test_video, test_author, test_date, test_comment2)
    print(f"Is this comment spam? {result2}")

# Save LoRA weights
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
model.save_pretrained_gguf("modelm0.4.0gguf", tokenizer, quantization_method="q4_k_m")
