import concurrent.futures
import time
from together import Together
from tqdm import tqdm
import os

class TogetherGenerator:
    def __init__(self, max_retries=1, retry_delay=2, max_workers=15, model_name="mistralai/Mistral-7B-Instruct-v0.3", default_system_prompt=None, evaluation = False):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.model_name = model_name
        self.default_system_prompt = default_system_prompt
        self.evaluation = evaluation
        self.together_client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def _generate_single_prompt(self, prompt, system_prompt=None):
        """Generate response for a single prompt with retries"""
        system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                if system_prompt is not None:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                response = self.together_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens= 512 if self.evaluation else 1024,
                    temperature=0 if self.evaluation else 1,
                    top_p=1 if self.evaluation else 0.9,
                    stop=["[/INST]", "</s>"],
                )
                content = response.choices[0].message.content
                
                if not self.evaluation:
                    
                    if "Revised Adversarial Prompt:" in content:
                        content = content.split("Revised Adversarial Prompt:")[-1].strip()
                    
                    if "Your revision:\n\n" in content:
                         content = content.split("Your revision:\n\n")[-1].strip()                       
                    
                    content = content.strip("*").strip("\n").strip("-").strip()
                
                if not self.evaluation:
                    delimiters = ["Given Simple Prompt:", "Given Revision Strategies:", "The adversarial prompt"]

                    for delimiter in delimiters:
                        if delimiter in content:
                            sections = content.split(delimiter)
                            return sections[0].strip()

                    return content.strip()

                return content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.max_retries} attempts. Error: {e}")
                    return None

    def generate_responses(self, prompts, system_prompt=None):
        """Generate responses for multiple prompts in parallel"""
        system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        outputs = [None] * len(prompts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._generate_single_prompt, prompt, system_prompt) for prompt in prompts]
            
            for i, future in enumerate(tqdm(futures, desc="Generating responses")):
                try:
                    outputs[i] = future.result(timeout=600)
                except concurrent.futures.TimeoutError:
                    print("Task timed out")
                    outputs[i] = None

        return outputs