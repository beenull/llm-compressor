from datasets import concatenate_datasets, load_dataset, load_from_disk

HF_DATASETS_CACHE = "/dataset/workspace/zhangl98/hf_cache/"

ds_vl = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", 
    "FigureQA(MathV360K)", 
    split="train[:128]",
    cache_dir= HF_DATASETS_CACHE
)


ds_text = load_dataset("hkust-nlp/deita-6k-v0", 
                       split="train[:128]",
                       cache_dir= HF_DATASETS_CACHE)


import pdb
pdb.set_trace()