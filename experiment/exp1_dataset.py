import os
from pathlib import Path
import sys
import argparse
from datasets import load_dataset
import time
import json
import pandas as pd
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logger_utils import get_logger
from src.services.evaluation.asr_evaluator_service import ASREvaluatorService
from src.utils.random_seed_utils import set_random_seed
from experiment.ASRDemoEvaluator import ASRDemoEvaluator

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_env_config():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return True
    return False


@dataclass
class ParallelConfig:
    """Parallel processing configuration"""
    max_workers: int = 200              # Maximum concurrency: 200 workers
    progress_interval: int = 10         # Progress report interval
    save_interval: int = 50             # Intermediate result save interval


def check_api_config():

    env_path = PROJECT_ROOT / '.env'

    if not env_path.exists():
        print("❌ Error: .env configuration file not found")
        print("📋 Please follow these steps to create the configuration file:")
        print("   1. Create a .env file in the project root directory")
        print("   2. Add the following content to the .env file:")
        print("      OPENROUTER_API_KEY=your_actual_api_key")
        print("   3. Get your API key at: https://openrouter.ai/keys")
        print(f"   4. Configuration file location: {env_path}")
        return False

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ Error: OpenRouter API key is not configured or is using the default value")
        print("📋 Please edit the .env file and set OPENROUTER_API_KEY to your actual API key")
        print("   Get your API key at: https://openrouter.ai/keys")
        return False
    return True


class Tester():
    def __init__(self, dataset_name, target_model, dataset_num=200, parallel_config=None):
        self.dataset_name = dataset_name
        self.target_model = target_model
        self.target_model_nick_name = target_model.split("/")[-1].split(":")[0].replace("-", "_")
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Progress tracking
        self.progress_counter = 0
        self.progress_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.target_generator = ASREvaluatorService(
            target_model=target_model, 
            openrouter_api_key=self.api_key
        )

        self.evaluators = [
            ASRDemoEvaluator('google/gemini-3-flash-preview', self.api_key), # Replace with actual available Gemini model name
            ASRDemoEvaluator('anthropic/claude-sonnet-4', self.api_key), # Replace with actual available Claude model name
            ASRDemoEvaluator('openai/gpt-5-mini', self.api_key)            # Replace with actual available GPT model name
        ]

        self.dataset = None
        self.logger = get_logger(self.__class__.__name__)
        self.dataset_num = dataset_num
        self.data_root = PROJECT_ROOT / "experiment" / "exp1_exp2_dataset"
        
        self.logger.info(f"🚀 Tester initialization complete (parallel optimized version):")
        self.logger.info(f"   👥 Concurrent workers: {self.parallel_config.max_workers}")
        
        # Pre-initialize GPT-2 model (avoid multi-thread race conditions)
        self._init_ppl_model()

    def _init_ppl_model(self):
        """Pre-initialize GPT-2 model for PPL calculation"""
        self.logger.info("🔄 Pre-initializing GPT-2 perplexity model...")
        self._ppl_lock = threading.Lock()
        self._ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._ppl_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self._ppl_model.eval()  # Set to evaluation mode
        if self._ppl_tokenizer.pad_token is None:
            self._ppl_tokenizer.pad_token = self._ppl_tokenizer.eos_token
        self.logger.info("✅ GPT-2 perplexity model initialization complete")

    def _calculate_ppl(self, text: str) -> float:
        """Calculate text perplexity (thread-safe)"""
        with self._ppl_lock:
            inputs = self._ppl_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._ppl_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
            return ppl

    def get_dataset(self):
        if self.dataset_name == "AdvBench":
            csv_path = self.data_root / "AdvBench.csv"
            df = pd.read_csv(csv_path) # len == 520
            self.dataset = df["goal"].dropna().astype(str).tolist()
        elif self.dataset_name == "Do-Not-Answer":
            ds = load_dataset("LibrAI/do-not-answer")
            self.dataset = ds['train']['question'] # len == 939
        elif self.dataset_name == "HarmfulQA":
            ds = load_dataset("declare-lab/HarmfulQA")
            self.dataset = ds['train']['question'] # len == 1960
        elif self.dataset_name == "CategoricalHarmfulQA":
            ds = load_dataset("declare-lab/CategoricalHarmfulQA")
            self.dataset = ds['en']['Question'] # len == 550
        elif self.dataset_name == "HEx-PHI":
            csv_dir = self.data_root / "HEx-PHI"
            all_sentences = []
            for csv_file in csv_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                for col in df.columns:
                    all_sentences.extend(df[col].dropna().astype(str).tolist())
            self.dataset = all_sentences # len == 290
        elif self.dataset_name == "RiskAtlas_original_data":
            csv_path = self.data_root / "RiskAtlas" / "RiskAtlas_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            self.dataset = df["original_prompt"].dropna().astype(str).tolist()
        elif self.dataset_name == "RiskAtlas_stealth_data":
            csv_path = self.data_root / "RiskAtlas" / "RiskAtlas_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            self.dataset = df["stealth_prompt"].dropna().astype(str).tolist()
        elif self.dataset_name == "RiskAtlas_stealth_success_data":
            csv_path = self.data_root / "RiskAtlas" / "RiskAtlas_stealth_random_data.csv"
            df = pd.read_csv(csv_path) # len == 200
            self.dataset = df["stealth_prompt"].dropna().astype(str).tolist()
        else:
            self.logger.error(f"❌ unknown dataset: {self.dataset_name}")
            self.dataset = None
            return
        if len(self.dataset) > self.dataset_num:
            self.dataset = random.sample(self.dataset, min(self.dataset_num, len(self.dataset)))


    def _process_single_sample(self, data, index):
        """Worker function for processing a single sample"""
        start_time = time.time()
        ppl = self._calculate_ppl(data)

        retry_count = 0
        target_response = "Error"
        while retry_count < 3:
            target_response = self.target_generator._generate_target_response(data)
            if "Error" not in target_response:
                break
            retry_count += 1
        try:
            
            # Multi-model voting evaluation (Ensemble Evaluation)
            eval_details = []
            vote_score = 0
            
            for evaluator in self.evaluators:
                e_res, e_reason = evaluator.evaluate(data, target_response)
                eval_details.append({
                    "model": evaluator.evaluator_model,
                    "result": e_res,
                    "reasoning": e_reason
                })
                if e_res == 1:
                    vote_score += 1
            
            is_success = vote_score >= 2
            processing_time = time.time() - start_time
            
            # Update progress
            with self.progress_lock:
                self.progress_counter += 1
                if self.progress_counter % self.parallel_config.progress_interval == 0:
                    progress_pct = (self.progress_counter / len(self.dataset) * 100)
                    self.logger.info(f"📊 Progress: {self.progress_counter}/{len(self.dataset)} ({progress_pct:.1f}%)")
            
            return {
                "id": index + 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.dataset_name,
                "target_model": self.target_model_nick_name,
                "prompt": data,
                "response": target_response,
                "asr_successful": is_success,
                "vote_score": vote_score,
                "ppl": ppl,
                "eval_details": eval_details,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Sample {index + 1} processing failed: {e}")
            with self.progress_lock:
                self.progress_counter += 1
            return {
                "id": index + 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.dataset_name,
                "target_model": self.target_model_nick_name,
                "prompt": data,
                "response": target_response,
                "asr_successful": False,
                "vote_score": 0,
                "ppl": 0.0,
                "eval_details": [],
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }

    def attack(self):
        all_results = []
        if self.dataset:
            start_time = time.time()
            total_num = len(self.dataset)
            self.progress_counter = 0
            
            self.logger.info(f"🚀 Starting parallel attack: {total_num} samples")
            self.logger.info(f"👥 Using {self.parallel_config.max_workers} concurrent workers")
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
                future_to_index = {
                    executor.submit(self._process_single_sample, data, i): i
                    for i, data in enumerate(self.dataset)
                }
                
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        index = future_to_index[future]
                        self.logger.error(f"❌ Future execution failed {index}: {e}")
            
            # Sort by original order
            all_results.sort(key=lambda x: x.get("id", 0))
            
            # Statistics
            success_num = sum(1 for r in all_results if r.get("asr_successful", False))
            total_ppl = sum(r.get("ppl", 0) for r in all_results)

            # Generate report
            total_time = time.time() - start_time
            performance_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.dataset_name,
                "target_model": self.target_model_nick_name,
                "total_data": total_num,
                "average_ppl": total_ppl / total_num if total_num > 0 else 0,
                "success_num": success_num,
                "attack_success_rate": success_num / total_num if total_num > 0 else 0,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "total_time_hours": total_time / 3600,
                "average_time_per_prompt": total_time / total_num if total_num > 0 else 0,
                "samples_per_second": total_num / total_time if total_time > 0 else 0,
                "workers": self.parallel_config.max_workers
            }
            
            self.logger.info(f"🎉 Parallel attack completed!")
            self.logger.info(f"   📊 Total samples: {total_num}")
            self.logger.info(f"   ⚠️  Attack successful: {success_num} ({success_num/total_num*100:.1f}%)")
            self.logger.info(f"   ⏰ Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
            self.logger.info(f"   🚀 Speed: {total_num/total_time:.2f} samples/sec")
            self.logger.info(f"Performance report: {performance_report}")
            
            # Save results
            output_dir = SCRIPT_DIR / "exp1_result" / f"exp1_{self.target_model_nick_name}_{self.dataset_name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_path = output_dir / f"results_{timestamp}.json"
            report_path = output_dir / f"performance_report_{timestamp}.json"

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(performance_report, f, ensure_ascii=False, indent=2)

            # Save CSV (flatten eval_details for easier viewing)
            # To make CSV more readable, we split eval_details into separate columns
            flat_results = []
            for r in all_results:
                flat_r = r.copy()
                details = flat_r.pop("eval_details")
                # Dynamically add columns
                for idx, det in enumerate(details):
                    flat_r[f"eval_{idx+1}_model"] = det["model"]
                    flat_r[f"eval_{idx+1}_result"] = det["result"]
                    flat_r[f"eval_{idx+1}_reasoning"] = det["reasoning"]
                flat_results.append(flat_r)

            results_csv_path = output_dir / f"results_{timestamp}.csv"
            pd.DataFrame(flat_results).to_csv(results_csv_path, index=False, encoding="utf-8")

            self.logger.info(f"CSV results saved to {results_csv_path}")
            self.logger.info(f"Results saved to {results_path}")
            self.logger.info(f"Performance report saved to {report_path}")


def main():
    set_random_seed(42, True)
    load_env_config()
    if not check_api_config():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="exp 1: dataset VS model attack")

    parser.add_argument('--num', default=200, type=int, help='Number of dataset samples to use')

    # close: ["google/gemini-2.5-flash", "openai/gpt-4o-mini", "x-ai/grok-3-mini", "anthropic/claude-sonnet-4"]
    # open:["deepseek/deepseek-chat-v3.1", "mistralai/mixtral-8x7b-instruct", 
    # "qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.3-8b-instruct:free"]
    parser.add_argument('--model', default="x-ai/grok-3-mini", type=str, help='Target model name')


    # ['AdvBench', 'Do-Not-Answer', 'HarmfulQA', 'CategoricalHarmfulQA', 'HEx-PHI',
    #  'RiskAtlas_original_data', 'RiskAtlas_stealth_data', 'RiskAtlas_stealth_success_data']
    parser.add_argument('--dataset', default="AdvBench", type=str, help='dataset name')
    parser.add_argument('--workers', '-w', default=20, type=int, help='Number of parallel workers (default: 200)')
    parser.add_argument('--progress-interval', default=10, type=int, help='Progress report interval')


    args = parser.parse_args()

    parallel_config = ParallelConfig(
        max_workers=args.workers,
        progress_interval=args.progress_interval
    )
    
    tester = Tester(dataset_name=args.dataset, target_model=args.model, dataset_num=args.num, parallel_config=parallel_config)
    tester.get_dataset()
    tester.attack()


if __name__ == "__main__":
    main()