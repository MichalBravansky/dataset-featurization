import json
from config import GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_PROMPT, client, NUM_EXAMPLE_SAMPLED, MODEL
import concurrent.futures
import time

class Generator:

    def get_feature(self, selected_string, strings, max_retries=3, retry_delay=2):
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": GENERATOR_USER_PROMPT.format(strings="\n\n---\n\n".join(strings), selected_string=selected_string)}
                    ],
                    temperature=0
                )

                return json.loads(completion.choices[0].message.content)["feature"]
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return []
    
    def analyze(self, df, comparison_column_name = None):

        def process(index):
            if comparison_column_name:
                strings = [df.loc[index, comparison_column_name]]
            else:
                strings = df.drop(index).sample(min(NUM_EXAMPLE_SAMPLED, len(df.index) - 1))["string"].to_list()

            return self.get_feature(df.loc[index, "string"], strings)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process, df.index))

        return [item for sublist in results for item in sublist]