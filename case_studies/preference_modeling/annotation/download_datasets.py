from datasets import load_dataset

df = load_dataset("dongyoung4091/shp-generated_flan_t5_large_flan_t5_large_zeroshot")["train"].to_pandas()[["prompt", "response"]]

indices = [i + j for i in range(0, len(df), 256) for j in range(16)]
df = df.iloc[indices].reset_index(drop=True)
df.to_parquet("data/shp-generated.parquet", index = False)

df = load_dataset("dongyoung4091/hh-generated_flan_t5_large_with_features2")["train"].to_pandas()[["prompt", "response"]]
df.to_parquet("data/hh-generated.parquet", index = False)

df = load_dataset("dongyoung4091/hh-rlhf_with_features")["train"].to_pandas()[["chosen", "rejected", "human", "assistant_chosen", "assistant_rejected", "labels"]]
df.to_parquet("data/hh-rlhf.parquet", index = False)

df = load_dataset("dongyoung4091/shp_with_features_20k")["train"].to_pandas()
df = df[['post_id', 'domain', 'upvote_ratio', 'history', 'c_root_id_A',
       'c_root_id_B', 'created_at_utc_A', 'created_at_utc_B', 'score_A',
       'score_B', 'human_ref_A', 'human_ref_B', 'labels', 'seconds_difference',
       'score_ratio']]

df.to_parquet("data/shp-with-features.parquet", index = False)

df = load_dataset("dongyoung4091/hh-rlhf_with_features")["test"].to_pandas()[["chosen", "rejected", "human", "assistant_chosen", "assistant_rejected", "labels"]]
df.to_parquet("data/hh-rlhf-test.parquet", index = False)

df = load_dataset("dongyoung4091/shp_with_features_20k")["test"].to_pandas()
df = df[['post_id', 'domain', 'upvote_ratio', 'history', 'c_root_id_A',
       'c_root_id_B', 'created_at_utc_A', 'created_at_utc_B', 'score_A',
       'score_B', 'human_ref_A', 'human_ref_B', 'labels', 'seconds_difference',
       'score_ratio']]

df.to_parquet("data/shp-with-features-test.parquet", index = False)