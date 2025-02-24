from config import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT, VERIFICATION_SPLIT, MODEL, client
import json
import pandas as pd
import random
import concurrent.futures
from tqdm import tqdm
import time
import math

class Verifier:

    def __init__(self):
        pass

    def verify(self, string, descriptions, max_retry=3, retry_delay=1):
        outputs = []
        
        for i in range(0, math.ceil(len(descriptions) / VERIFICATION_SPLIT)):
            features = "\n".join(descriptions[i*VERIFICATION_SPLIT:i*VERIFICATION_SPLIT + VERIFICATION_SPLIT])
            prompt = VERIFICATION_USER_PROMPT.format(text=string, features=features)
            
            retries = 0
            while retries < max_retry:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    completion_dict = json.loads(completion.choices[0].message.content)
                    break
                except Exception as e:
                    print(e)
                    retries += 1
                    if retries == max_retry:
                        completion_dict = {}
                        break
                    time.sleep(retry_delay)
            
            incremented_dict = {descriptions[i*VERIFICATION_SPLIT + k]: True if v == "Y" else False for k, v in enumerate(completion_dict.values()) if k < VERIFICATION_SPLIT and i*VERIFICATION_SPLIT + k < len(descriptions)}
            
            outputs.extend(incremented_dict.items())

        final_output = dict(outputs)

        final_output["string"] = string

        return final_output
    

    def process(self, strings: list, descriptions: list, max_workers=15):
        outputs = []

        print(len(descriptions))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.verify, string, descriptions)
                for string in strings
            ]

            for future in tqdm(concurrent.futures.as_completed(futures)):
                try:
                    output = future.result(timeout=60)
                    outputs.append(output)
                except concurrent.futures.TimeoutError:
                    print(f"Task timed out for futxure {future}")
                    outputs.append(None)


        df = pd.DataFrame(outputs)

        df = pd.merge(pd.DataFrame(strings, columns = ["string"]),df.drop_duplicates(subset=["string"]), on="string", how="left")
        df = df[list(df.columns)[1:] + ["string"]]
        df = df.dropna(how='all', subset=list(df.columns[:-1]))

        return df