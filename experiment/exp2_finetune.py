from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import random
import pandas as pd
from pathlib import Path
import sys
from unsloth import is_bfloat16_supported
import argparse
import torch
import os.path as osp

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.services.evaluation.asr_evaluator_service import ASREvaluatorService
from src.utils.random_seed_utils import set_random_seed

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

class FinetuneExecuter:
    def __init__(
        self, 
        finetune_dataset_name="AdvBench",
        finetune_model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        finetune_dataset_num=200,
    ):
        # unsloth/Meta-Llama-3.1-8B-bnb-4bit
        self.refusal_examples = [
            "I'm sorry, but I cannot provide assistance with that request.",
            "I’m unable to help with that as it involves harmful content.",
            "Apologies, but I cannot generate that kind of response.",
            "I must decline to respond to that prompt for safety reasons.",
            "I cannot fulfill this request as it may cause harm.",
            "Sorry, I cannot provide information on that topic.",
            "I’m not able to complete that request because it’s unsafe.",
            "I must refuse to answer this question to protect users.",
            "I cannot give guidance on that matter due to safety concerns.",
            "Apologies, but I am unable to respond to harmful prompts.",
            "I cannot assist with this request because it involves dangerous content.",
            "I’m here to provide safe guidance and cannot answer that.",
            "Sorry, I cannot generate content that could be harmful.",
            "I’m unable to provide instructions related to that request.",
            "I must decline because this prompt is unsafe to respond to.",
            "I cannot produce output for this prompt as it may be harmful.",
            "Apologies, I cannot engage with this kind of request.",
            "I cannot respond to this prompt as it violates safety rules.",
            "I’m sorry, but I cannot comply with this instruction.",
            "I cannot assist with this request due to potential risks."
        ]

        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

        self.max_seq_length = 2048 
        self.dtype = None
        self.load_in_4bit = True

        self.raw_data_root = SCRIPT_DIR / "exp1_exp2_dataset"
        self.dataset_name = finetune_dataset_name
        self.finetune_dataset_num = finetune_dataset_num
        self.finetune_dataset = None

        self.model_name = finetune_model_name
        if self.model_name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit":
            self.model_path = osp.join(PROJECT_ROOT, "models", "Llama-3.1-8b-bnb-4bit-finetune") # instruct finetune model

        self.model = None
        self.tokenizer = None
        self.EOS_TOKEN = None

    def load_finetune_dataset(self):
        if self.dataset_name == "AdvBench":
            csv_path = self.raw_data_root / "AdvBench.csv"
            df = pd.read_csv(csv_path) # len == 520
            raw_dataset = df["goal"].dropna().astype(str).tolist()
        elif self.dataset_name == "Do-Not-Answer":
            ds = load_dataset("LibrAI/do-not-answer")
            raw_dataset = ds['train']['question'] # len == 939
        elif self.dataset_name == "HarmfulQA":
            ds = load_dataset("declare-lab/HarmfulQA")
            raw_dataset = ds['train']['question'] # len == 1960
        elif self.dataset_name == "CategoricalHarmfulQA":
            ds = load_dataset("declare-lab/CategoricalHarmfulQA")
            raw_dataset = ds['en']['Question'] # len == 550
        elif self.dataset_name == "HEx-PHI":
            csv_dir = self.raw_data_root / "HEx-PHI"
            all_sentences = []
            for csv_file in csv_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                for col in df.columns:
                    all_sentences.extend(df[col].dropna().astype(str).tolist())
            raw_dataset = all_sentences # len == 290
        elif self.dataset_name == "RiskAtlas_original_data":
            csv_path = self.raw_data_root / "RiskAtlas" / "RiskAtlas_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            raw_dataset = df["original_prompt"].dropna().astype(str).tolist()
        elif self.dataset_name == "RiskAtlas_stealth_data":
            csv_path = self.raw_data_root / "RiskAtlas" / "RiskAtlas_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            raw_dataset = df["stealth_prompt"].dropna().astype(str).tolist()
        elif self.dataset_name == "RiskAtlas_stealth_success_data":
            csv_path = self.raw_data_root / "RiskAtlas" / "RiskAtlas_stealth_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            raw_dataset = df["stealth_prompt"].dropna().astype(str).tolist()
        else:
            self.logger.error(f"❌ unknown dataset: {self.dataset_name}")
            self.finetune_dataset = None
            return

        if len(raw_dataset) > self.finetune_dataset_num:
            raw_dataset = random.sample(raw_dataset, min(self.finetune_dataset_num, len(raw_dataset)))

        texts = []
        for instruction in raw_dataset:
            response = random.choice(self.refusal_examples)
            text = self.alpaca_prompt.format(instruction, "", response) + self.EOS_TOKEN
            texts.append(text)
        self.finetune_dataset = Dataset.from_dict({"text": texts})

        # Print the first 3 finetuning samples as examples
        print("First 3 finetuning sample examples:")
        for i, sample in enumerate(self.finetune_dataset["text"][:3]):
            print(f"Example {i+1}:\n{sample}\n")


    def load_model(self):
        raw_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_path,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            raw_model,
            r = 64, 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 128,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 42,
            use_rslora = True,
            loftq_config = None, # And LoftQ
        )

        self.EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN

    def initialize(self):
        self.load_model()
        self.load_finetune_dataset()
        
        prompt = self.alpaca_prompt.format("Who are you?", "", "")

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
    
    def finetune(self):
        if not self.model or not self.tokenizer or not self.finetune_dataset:
            print("❌ Model or dataset not properly loaded, cannot proceed with finetuning.")
            return
        print("✅ Starting finetuning...")
        args = TrainingArguments(
            per_device_train_batch_size = 2,  # Reduce batch size to accommodate larger LoRA rank
            gradient_accumulation_steps = 8,  # Increase gradient accumulation steps accordingly
            warmup_steps = 20,  # Increase warmup steps for smoother training start
            # num_train_epochs = 1, # Use more epochs for better convergence
            num_train_epochs = 3, # Use more epochs for better convergence
            # max_steps = 60,  # Use epochs instead of fixed steps
            learning_rate = 2e-6,  # Reasonable learning rate
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",  # Use cosine scheduler for better convergence
            seed = 42,
            output_dir = str(SCRIPT_DIR / "finetune_model_checkpoint" / f"{self.dataset_name}_{self.model_name.split('/')[-1]}"),
            report_to = "none", # Use this for WandB etc.
        )

        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.finetune_dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = args
        )
        trainer_stats = trainer.train()

        prompt = self.alpaca_prompt.format("Who are you?", "", "")

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

        self.model.save_pretrained(str(SCRIPT_DIR / "finetune_model" / f"{self.dataset_name}_{self.model_name.split('/')[-1]}"))
        self.tokenizer.save_pretrained(str(SCRIPT_DIR / "finetune_model" / f"{self.dataset_name}_{self.model_name.split('/')[-1]}"))


def main():
    set_random_seed(42, True)

    parser = argparse.ArgumentParser(description="exp 2: fintune model")

    parser.add_argument('--num', default=200, type=int, help='Number of dataset samples to use')

    # ['AdvBench', 'Do-Not-Answer', 'HarmfulQA', 'CategoricalHarmfulQA', 'HEx-PHI',
    #  'RiskAtlas_original_data', 'RiskAtlas_stealth_data', 'RiskAtlas_stealth_success_data']
    parser.add_argument('--finetune_dataset', default="RiskAtlas_stealth_success_data", type=str, help='dataset name')


    args = parser.parse_args()

    executer = FinetuneExecuter(finetune_dataset_name=args.finetune_dataset, finetune_dataset_num=args.num)
    executer.initialize()
    executer.finetune()

if __name__ == "__main__":
    main()
