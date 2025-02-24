import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import numpy as np
import logging
import pickle
from tqdm import tqdm
from transformers import DynamicCache
import copy
from config import prompt_template
from collections import defaultdict

class Evaluator:

    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload" 
        )
        self.model = self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.end_token = 128007
        self.batch_size = batch_size

    def init_cached_prompts(self, string_id):
        self.prompts = string_id.apply(lambda row: self.get_prompt(row, [], description=None, sample=True), axis = 1).tolist()

    def _compute_cache(self, cache_id):
        prompt = self.prompts[cache_id]
        encodings = self.tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"][:, :-1].to(self.device)
        attention_mask = encodings["attention_mask"][:, :-1].to(self.device)
        cache_length = input_ids.size(1)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            cache = outputs.past_key_values
        return cache, attention_mask, cache_length

    def _expand_past_key_values(self, past_key_values, batch_size):
        expanded_past = []
        for layer in past_key_values:
            key, value = layer
            key = key.expand(batch_size, -1, -1, -1).contiguous()
            value = value.expand(batch_size, -1, -1, -1).contiguous()
            expanded_past.append((key, value))
        return expanded_past


    def calculate_perplexity(self, cache_ids, data):
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)

        data_encodings = self.tokenizer(
            data,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors=None,
        )
        data_input_ids = data_encodings['input_ids']

        cache_id_to_indices = defaultdict(list)
        for idx, cache_id in enumerate(cache_ids):
            cache_id_to_indices[cache_id].append(idx)

        ppls = [0] * len(data)
        unique_cache_ids = list(set(cache_ids))

        for cache_id in tqdm(unique_cache_ids):
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cache, cache_attention_mask, cache_length = self._compute_cache(cache_id)

            indices = cache_id_to_indices[cache_id]
            for start_index in range(0, len(indices), self.batch_size):
                end_index = min(start_index + self.batch_size, len(indices))
                batch_indices = indices[start_index:end_index]
                batch_data_input_ids = [data_input_ids[i] for i in batch_indices]

                sliced_input_ids = []
                for input_ids in batch_data_input_ids:
                    input_ids = input_ids[cache_length:]
                    sliced_input_ids.append(torch.tensor(input_ids, dtype=torch.long))

                padded_batch = self.tokenizer.pad(
                    {"input_ids": sliced_input_ids},
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = padded_batch["input_ids"].to(self.device)
                current_attention_mask = padded_batch["attention_mask"].to(self.device)

                batch_size_current = len(batch_indices)
                cache_attention_mask_expanded = cache_attention_mask.repeat(batch_size_current, 1)
                cache_expanded = self._expand_past_key_values(cache, batch_size_current)

                attn_mask = torch.cat([cache_attention_mask_expanded.to(self.device), current_attention_mask], dim=1)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids,
                        attention_mask=attn_mask,
                        past_key_values=cache_expanded,
                    )
                logits = outputs.logits

                colon_token_id = torch.tensor([self.end_token]).to(self.device)
                colon_indices = []

                for i in range(input_ids.size(0)):
                    colon_indices_temp = (input_ids[i] == colon_token_id).nonzero(as_tuple=True)[0]
                    if len(colon_indices_temp) > 0:
                        last_colon_index = colon_indices_temp[-1].item()
                        colon_indices.append(last_colon_index)
                    else:
                        colon_indices.append(None)

                perplexity_batch = []
                for i in range(input_ids.size(0)):
                    colon_index = colon_indices[i] if i < len(colon_indices) else -1
                    
                    if colon_index >= 0:
                        shift_logits = logits[i, colon_index:-1, :].contiguous()
                        shift_labels = input_ids[i, colon_index+1:].contiguous()
                        shift_attention_mask = current_attention_mask[i, colon_index+1:].contiguous()

                        if shift_labels.size(0) > 0:
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            perplexity = torch.exp2(
                                (loss * shift_attention_mask).sum() / shift_attention_mask.sum()
                            )
                            perplexity_batch.append(perplexity.item())
                        else:
                            logging.warning(f"No end token for sample {i} in batch. Skipping.")
                    else:
                        logging.warning(f"No end token for sample {i} in batch. Skipping.")

                for idx, perplexity in zip(batch_indices, perplexity_batch):
                    ppls[idx] = perplexity

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    
    def get_prompt(self, row, feature_names, description = None, sample = False):

        if feature_names:
            features = "\n".join([feature for feature in feature_names if row[feature + "_property"] == True])
            
            if features!="":
                features = "\n" + features
        else:
            features = ""

        if description:
            description = "\n" + description
        else:
            description = ""

        if description != "":
            description = description
            features = features
        
        elif features != "":
            features = features
        else:
            features = ""

        system_message, user_message, assistant_message = prompt_template(row, features + description)
        chat_history = [{"role": "system", "content": system_message}]
        chat_history += [{"role": "user", "content": user_message}]

        if not sample:
            chat_history += [{"role": "assistant", "content": assistant_message}]

        
        prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False)

        # Llama automatically adds the date to the system prompt, we remove it here
        if "llama" in self.model.config.model_type.lower():
            prompt_split = prompt.split("\n\n")
            prompt = "\n\n".join([prompt_split[0]] + prompt_split[2:])

        return prompt


    def get_perplexities(self, string_ids, description, string_df, feature_names, no_string=False):
        series = string_df.apply(lambda x: self.get_prompt(x, feature_names, description), axis=1)
        batches = series.tolist()
        perplexities = self.calculate_perplexity(string_ids, batches)
        return {string_df.index[i]: perplexities["perplexities"][i] for i in range(len(perplexities["perplexities"]))}


    def evaluate(self, classification_df, string_df, string_ids, feature_names = []):

        empty_perplexities = self.get_perplexities(string_ids, None, string_df, feature_names)
    
        self._string_df = string_df

        properties = []

        batches_to_process = []
        ids_to_process = []

        for description in tqdm(classification_df.columns[:-1]):
            yes_ids = classification_df[classification_df[description] == True][description].index.tolist()
            if len(yes_ids) == 0:
                continue

            batches_to_process += string_df.loc[yes_ids].apply(lambda x: self.get_prompt(x, feature_names, description), axis=1).tolist()
            ids_to_process += [string_ids[i] for i in string_df.loc[yes_ids].index]

        batch_perplexity = self.calculate_perplexity(ids_to_process, batches_to_process)["perplexities"]
        perplexity_id = 0

        for description in tqdm(classification_df.columns[:-1]):
            yes_ids = classification_df[classification_df[description] == True][description].index
            yes_length = len(yes_ids)

            no_ids = classification_df[classification_df[description] == False][description].index

            if yes_length > 0:
                yes_perplexities = {yes_id: batch_perplexity[id] for id, yes_id in enumerate(yes_ids, perplexity_id)}
                perplexity_id += yes_length
            else:
                yes_perplexities = {}
                
            no_perplexities = {id: empty_perplexities[id] for id in no_ids}

            properties.append({**yes_perplexities, **no_perplexities})

        temp_df = pd.DataFrame(properties)
        temp_df = temp_df[sorted(temp_df.columns)]
        temp_df["description"] = classification_df.columns[:len(temp_df.index)]
        temp_df.columns = classification_df.index.tolist() + ["description"]
        temp_df = temp_df.transpose()
        temp_df.columns = temp_df.loc[["description"]].to_numpy().tolist()[0]
        temp_df.drop(["description"], inplace= True)
        temp_df["empty"] = empty_perplexities.values()

        temp_df.fillna(1000, inplace = True)

        return temp_df