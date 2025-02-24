import concurrent.futures
import time
from openai import OpenAI
from tqdm import tqdm
import os

class OpenAIGenerator:
    def __init__(self, model_name, max_retries=3, retry_delay=2, max_workers=15):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.model = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
    def _generate_single_prompt(self, prompt):
        """Generate response for a single prompt with retries"""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature = 0,
                    top_p = 1,
                    max_tokens = 512
                )
                return completion.choices[0].message.content
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.max_retries} attempts. Error: {e}")
                    return None

    def generate_responses(self, prompts):
        """Generate responses for multiple prompts in parallel"""
        outputs = [None] * len(prompts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._generate_single_prompt, prompt) for prompt in prompts]
            
            for i, future in enumerate(tqdm(futures, desc="Generating responses")):
                try:
                    outputs[i] = future.result(timeout=600)
                except concurrent.futures.TimeoutError:
                    print("Task timed out")
                    outputs[i] = None

        return outputs