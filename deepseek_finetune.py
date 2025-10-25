# deepseek_finetune.py
import os
import json
import torch, gc
gc.collect()
torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb

# =====================
# 1️⃣  Setup environment
# =====================
os.environ["WANDB_PROJECT"] = "deepseek-finetune-local"
os.environ["WANDB_LOG_MODEL"] = "false"  # optional
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: login to wandb (comment out if you don’t use it)
# wandb.login(key="YOUR_WANDB_API_KEY")

# =====================
# 2️⃣  Define model
# =====================
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)


# =====================
# 3️⃣  Create dataset
# =====================
samples = [
    {
        "prompt": "Question 1: What is the first step in assessing a vintage camera's condition?",
        "completion": "Answer 1: Begin by visually inspecting the camera's exterior for rust, cracks, and loose components."
    },
    {
        "prompt": "Question 2: How do you properly clean the lens of an old film camera?",
        "completion": "Answer 2: Use a microfiber cloth and a small amount of lens cleaning solution, avoiding harsh chemicals."
    },
    {
        "prompt": "Question 3: What common issues affect the shutter mechanism in vintage cameras?",
        "completion": "Answer 3: Common issues include worn springs, sticky lubrication, and misaligned curtains."
    },
    {
        "prompt": "Question 4: How can you safely disassemble a vintage camera for restoration?",
        "completion": "Answer 4: Document every step with photos, use a precision screwdriver set, and keep track of small parts."
    },
    {
        "prompt": "Question 5: What type of lubricant is recommended for restoring old camera mechanisms?",
        "completion": "Answer 5: A light synthetic oil, such as sewing machine oil, is typically safe and effective."
    },
    {
        "prompt": "Question 6: How do you test a restored vintage camera's functionality?",
        "completion": "Answer 6: Test the shutter speed, aperture settings, and focus accuracy under various lighting conditions."
    },
    {
        "prompt": "Question 7: What essential tools are needed for vintage camera restoration?",
        "completion": "Answer 7: Key tools include a precision screwdriver set, tweezers, a magnifying glass, and specialized cleaning solutions."
    },
    {
        "prompt": "Question 8: How do you remove mold from the interior of a vintage camera?",
        "completion": "Answer 8: Gently remove mold with a soft brush and isopropyl alcohol, taking care not to damage sensitive parts."
    },
    {
        "prompt": "Question 9: What methods can be used to repair a misaligned viewfinder?",
        "completion": "Answer 9: Carefully adjust the viewfinder by loosening its mount, repositioning it, and then securing it back."
    },
    {
        "prompt": "Question 10: What are the risks of using modern lubricants on vintage cameras?",
        "completion": "Answer 10: Modern lubricants may cause residue buildup or react adversely with the camera's original materials."
    },
    {
        "prompt": "Question 11: How do you restore the leather casing of a vintage camera?",
        "completion": "Answer 11: Clean the leather with a dedicated leather cleaner, apply a conditioner, and repair minor tears using leather glue."
    },
    {
        "prompt": "Question 12: What precautions should be taken when handling antique camera electronics?",
        "completion": "Answer 12: Always work with the camera unplugged, and handle electronic components with anti-static tools."
    },
    {
        "prompt": "Question 13: How do you prevent further deterioration during restoration?",
        "completion": "Answer 13: Work in a clean, controlled environment and use anti-static mats to reduce dust and static discharge."
    },
    {
        "prompt": "Question 14: What is the recommended storage method for restored vintage cameras?",
        "completion": "Answer 14: Store in a cool, dry place using padded cases or cabinets to minimize exposure to dust and humidity."
    },
    {
        "prompt": "Question 15: How do you safely remove rust from metal camera parts?",
        "completion": "Answer 15: Remove rust using fine abrasive tools or rust removers, then apply a rust inhibitor to prevent recurrence."
    },
    {
        "prompt": "Question 16: What challenges might you face when sourcing parts for vintage cameras?",
        "completion": "Answer 16: Sourcing can be difficult due to scarcity; sometimes parts must be substituted from similar models or custom fabricated."
    },
    {
        "prompt": "Question 17: How can you verify the authenticity of a vintage camera model?",
        "completion": "Answer 17: Check serial numbers, manufacturer markings, and compare with historical records or reference guides."
    },
    {
        "prompt": "Question 18: What precautions are necessary when cleaning camera film compartments?",
        "completion": "Answer 18: Avoid scratching film surfaces by using soft brushes and gentle cleaning agents, and never use aggressive solvents."
    },
    {
        "prompt": "Question 19: How do you recalibrate exposure settings on an old camera?",
        "completion": "Answer 19: Use a calibrated light meter to adjust the aperture and shutter speed settings for accurate exposure."
    },
    {
        "prompt": "Question 20: What benefits do professional restoration services offer?",
        "completion": "Answer 20: Professionals bring specialized expertise and tools that ensure the camera is restored without compromising its historical value."
    },
    {
        "prompt": "Question 21: How do you address sticky shutter curtains?",
        "completion": "Answer 21: Clean the mechanism with appropriate solvents and lubricants to restore smooth and consistent movement."
    },
    {
        "prompt": "Question 22: What signs indicate a faulty mirror mechanism in vintage SLR cameras?",
        "completion": "Answer 22: Delays or erratic movement of the mirror are common indicators of worn springs or misalignment."
    },
    {
        "prompt": "Question 23: How do you fix a jammed film advance mechanism?",
        "completion": "Answer 23: Inspect for debris, clean the gears, and carefully lubricate moving parts to free the mechanism."
    },
    {
        "prompt": "Question 24: What techniques are used to repair cracked camera bodies?",
        "completion": "Answer 24: Depending on the material, use specialized plastic adhesives or metal welding techniques for durable repairs."
    },
    {
        "prompt": "Question 25: How can UV light damage vintage cameras?",
        "completion": "Answer 25: UV exposure can degrade plastics and fade printed labels; thus, cameras should be stored away from direct sunlight."
    },
    {
        "prompt": "Question 26: What effect does humidity have on vintage camera components?",
        "completion": "Answer 26: High humidity can promote rust and mold, while extremely low humidity may cause plastics to become brittle."
    },
    {
        "prompt": "Question 27: How do you safely clean a vintage rangefinder's viewfinder?",
        "completion": "Answer 27: Use a soft, lint-free cloth dampened with isopropyl alcohol, and avoid applying too much pressure on the optics."
    },
    {
        "prompt": "Question 28: What are indicators that a vintage camera might need professional maintenance?",
        "completion": "Answer 28: Persistent malfunctions, unusual noises, or visible structural damage are signs that professional help may be required."
    },
    {
        "prompt": "Question 29: How do you approach cleaning delicate internal gears in an old camera?",
        "completion": "Answer 29: Use compressed air and fine brushes to carefully remove dust without disturbing the gear alignment."
    },
    {
        "prompt": "Question 30: What techniques help in restoring faded camera markings?",
        "completion": "Answer 30: Retouch with archival-quality inks or decals, ensuring that any restoration is reversible and preserves authenticity."
    },
    {
        "prompt": "Question 31: How can you repair loose or missing screws in a vintage camera?",
        "completion": "Answer 31: Replace them with screws of similar size or use thread-locking compounds to secure them in place."
    },
    {
        "prompt": "Question 32: What steps should be taken if a vintage camera’s light meter malfunctions?",
        "completion": "Answer 32: First, calibrate the meter with a known light source; if problems persist, consult a specialist."
    },
    {
        "prompt": "Question 33: How do you resolve issues with old camera film spindles?",
        "completion": "Answer 33: Clean the spindle thoroughly and apply a small amount of the recommended lubricant to ensure smooth rotation."
    },
    {
        "prompt": "Question 34: Why is it important to document the restoration process?",
        "completion": "Answer 34: Documentation ensures reproducibility, aids troubleshooting, and preserves the camera’s restoration history."
    },
    {
        "prompt": "Question 35: How can you mitigate the risk of static electricity during restoration work?",
        "completion": "Answer 35: Work on an anti-static mat and use anti-static tools to protect sensitive electronic and mechanical components."
    },
    {
        "prompt": "Question 36: What cleaning solution is best for delicate camera electronics?",
        "completion": "Answer 36: A diluted solution of distilled water with isopropyl alcohol is typically safe for cleaning electronics."
    },
    {
        "prompt": "Question 37: How do you restore a deteriorated camera strap?",
        "completion": "Answer 37: Either repair the existing strap using specialized adhesives or replace it with a high-quality leather or fabric alternative."
    },
    {
        "prompt": "Question 38: What are the key differences between restoring a vintage camera and a modern digital camera?",
        "completion": "Answer 38: Vintage cameras rely on mechanical parts that require manual adjustments, whereas digital cameras have electronic systems and firmware."
    },
    {
        "prompt": "Question 39: How do you evaluate the historical value of a vintage camera?",
        "completion": "Answer 39: Consider the camera’s rarity, production year, design, and its role in the evolution of photography."
    },
    {
        "prompt": "Question 40: What steps help prevent condensation inside a vintage camera body?",
        "completion": "Answer 40: Allow the camera to acclimate gradually to temperature changes and use silica gel packs in storage."
    },
    {
        "prompt": "Question 41: How should vintage camera film rolls be maintained?",
        "completion": "Answer 41: Store them in a cool, dry place and handle them with gloves to prevent oils from your skin transferring to the film."
    },
    {
        "prompt": "Question 42: What are the risks of using ultrasonic cleaners on vintage cameras?",
        "completion": "Answer 42: Ultrasonic cleaning may damage delicate mechanical parts or dislodge small components crucial for proper operation."
    },
    {
        "prompt": "Question 43: How can you improve the performance of an aging camera shutter?",
        "completion": "Answer 43: Replace worn components, adjust spring tensions, and ensure the mechanism is properly cleaned and lubricated."
    },
    {
        "prompt": "Question 44: What environmental factors should be avoided during the restoration process?",
        "completion": "Answer 44: Avoid working in dusty, extremely humid, or high-temperature environments that can further damage the camera."
    },
    {
        "prompt": "Question 45: How do you safely dispose of outdated lubricants and cleaning chemicals?",
        "completion": "Answer 45: Follow local hazardous waste disposal guidelines to ensure that chemicals are discarded in an environmentally safe manner."
    },
    {
        "prompt": "Question 46: What are some signs that a vintage camera component is beyond repair?",
        "completion": "Answer 46: Extensive corrosion, irreparable cracks, or broken mechanisms indicate that a component may need to be replaced rather than restored."
    },
    {
        "prompt": "Question 47: How can digital tools assist in the restoration of vintage cameras?",
        "completion": "Answer 47: Digital imaging and measurement tools can document the process and help achieve precise adjustments."
    },
    {
        "prompt": "Question 48: What maintenance routine is recommended after a vintage camera has been restored?",
        "completion": "Answer 48: Regular cleaning, periodic lubrication, and proper storage will help maintain the camera’s condition over time."
    },
    {
        "prompt": "Question 49: How do you ensure that restoration efforts do not compromise a camera's originality?",
        "completion": "Answer 49: Use reversible restoration techniques and authentic replacement parts where possible to preserve the camera's historical integrity."
    },
    {
        "prompt": "Question 50: What is the final step after completing a vintage camera restoration?",
        "completion": "Answer 50: Thoroughly test the camera’s functionality, document the restoration process, and store the camera properly to safeguard its restored state."
    }
]

dataset_path = "niche_dataset.jsonl"
with open(dataset_path, "w", encoding="utf-8") as f:
    for sample in samples:
        json_line = json.dumps(sample, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"✅ Created dataset file: {dataset_path}")

# =====================
# 4️⃣  Load and tokenize
# =====================
dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
split = dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

def tokenize_function(examples):
    combined_texts = [f"{p}\n{c}" for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(combined_texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# =====================
# 5️⃣  PEFT (LoRA) config
# =====================
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================
# 6️⃣  Training setup
# =====================
training_args = TrainingArguments(
    output_dir="./deepseek_finetuned",
    num_train_epochs=3,  # reduce for local run
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=50,
    # evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    report_to="wandb",  # or [] if you don’t want to log to wandb
    run_name="DeepSeek_FineTuning_Local",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# =====================
# 7️⃣  Train model
# =====================
trainer.train()

# =====================
# 8️⃣  Save locally
# =====================
save_dir = "./deepseek_finetuned"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Model saved to {save_dir}")

# =====================
# 9️⃣  Merge LoRA weights (optional)
# =====================
from peft import PeftModel
import torch, gc

gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype=torch.float16,
).to("cuda")

peft_model = PeftModel.from_pretrained(base_model, save_dir)
merged_model = peft_model.merge_and_unload()

final_dir = "./deepseek_finetuned_full"
os.makedirs(final_dir, exist_ok=True)
merged_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"✅ Merged full model saved to {final_dir}")


final_dir = "./deepseek_finetuned_full"
os.makedirs(final_dir, exist_ok=True)
merged_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"✅ Merged full model saved to {final_dir}")

# =====================
# 🔟  Test generation (manual)
# =====================
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "Best place to visit on Canada Day?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # move input to GPU
outputs = merged_model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)
print("\nGenerated text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
