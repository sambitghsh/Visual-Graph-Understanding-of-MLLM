import torch
import pandas as pd
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import os
import json
from tqdm import tqdm

 
# Load model and processor
model_id = "llava-hf/llava-1.5-13b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir="/dccstor/mkesar/nltogql/temp_files_exp/Llava_13b", torch_dtype=torch.float16).eval()
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
processor = LlavaProcessor.from_pretrained(model_id, cache_dir="/dccstor/mkesar/nltogql/temp_files_exp/Llava_13b")
 
 
excel_path = '../DATA_SUBSET/Vision_Graph_5_subset.xlsx'
 
 
 
# Prompts
prompts = [
    "USER: <image> Can you describe the graph in the given image? Pleasne inlcude number of node, description of edges and whatever you are able to notice in the graph.\nASSISTANT:",
    # "USER: <image> What is in the image, a graph or a line or a circle or a pneumonoultramicroscopicsilicovolcanoconiosis? Please answer in one word.\nASSISTANT:",
    # "USER: <image> Can you see the graph? Please answer either yes or no.\nASSISTANT:",
    # "USER: <image> Is the graph directed? Please answer either yes or no.\nASSISTANT:",
    # "USER: <image> Is the graph weighted? Please answer either yes or no.\nASSISTANT:",
    # "USER: <image> How many nodes are there in the graph?\nASSISTANT:",
    # "USER: <image> How many edges are there in the graph?\nASSISTANT:",
    # "USER: <image> Can you give a sorted list of the nodes?\nASSISTANT: The sorted list of nodes are "
]
 
def run_llava_and_save_jsonl_batch(excel_path, output_jsonl="llava_output.jsonl", batch_size=4):
    df = pd.read_excel(excel_path)
    df.drop(df.columns[df.columns.str.contains('Unnamed: 0.1', case=False)], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('Unnamed: 0', case=False)], axis=1, inplace=True)
    df = df.rename_axis('id')
    df = df.reset_index()
    results = []
 
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Batch processing"):
        batch = df.iloc[start_idx:start_idx+batch_size]
        images, image_ids, image_paths = [], [], []
 
        for _, row in batch.iterrows():
            try:
                image = Image.open(os.path.join('../DATA_SUBSET/image_subset', row["image"])).convert("RGB")
                images.append(image)
                image_ids.append(row["id"])
                image_paths.append(os.path.join('../DATA_SUBSET/image_subset', row["image"]))
 
            except Exception as e:
                results.append({
                    "id": row["id"],
                    "image": row["image"],
                    "responses": {},
                    "error": f"Failed to load image: {e}"
                })
 
        for i, prompt in enumerate(prompts):
            try:
                inputs = processor([prompt]*len(images), images, return_tensors="pt", padding=True).to(model.device, torch.float16)
                outputs = model.generate(**inputs, max_new_tokens=1000)
 
                for j, output in enumerate(outputs):
                    decoded = processor.decode(output, skip_special_tokens=True)
                    entry_idx = next((idx for idx, r in enumerate(results) if r.get("id") == image_ids[j]), None)
                    print(decoded)
                    if entry_idx is None:
                        results.append({
                            "id": image_ids[j],
                            "image": image_paths[j],
                            "responses": {f"prompt{i}": {
                                "prompt": prompt,
                                "response": decoded
                            }}
                        })
                    else:
                        results[entry_idx]["responses"][f"prompt{i}"] = {
                            "prompt": prompt,
                            "response": decoded
                        }
            except Exception as e:
                for j in range(len(images)):
                    results.append({
                        "id": image_ids[j],
                        "image": image_paths[j],
                        "responses": {
                            f"prompt{i}": {
                                "prompt": prompt,
                                "error": str(e)
                            }
                        }
                    })
 
    # Write to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
 
    print(f"\n Batch inference complete. Results saved to '{output_jsonl}'.")
 
# Example usage
run_llava_and_save_jsonl_batch(excel_path, "../DATA_SUBSET/llava_results_batch_13b_lp2.jsonl", batch_size=4)
 
