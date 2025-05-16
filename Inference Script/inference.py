import argparse
import os
import zipfile
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from huggingface_hub import login

ONEDRIVE_URL = "https://iiitbac-my.sharepoint.com/:u:/g/personal/abhik_kumar_iiitb_ac_in/EWDjbbKyGihIspF38g___PwBHliaFQk7nSdT4gtH3v01Zw?e=berUad"

def fetch_adapter_from_onedrive(url: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "checkpoint-30000.zip")
    if not os.path.isfile(archive_path):
        dl_url = url + ("&download=1" if "?" in url else "?download=1")
        resp = requests.get(dl_url, allow_redirects=True)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):
            raise RuntimeError(f"Expected ZIP file, got {resp.headers.get('Content-Type')}")
        with open(archive_path, "wb") as out:
            out.write(resp.content)
    extract_path = os.path.join(target_dir, "extracted")
    if not os.path.isdir(extract_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_path)
    return extract_path


def build_model(adapter_folder: str):
    login("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")
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16,
        device_map="auto",
        revision="float16"
    )
    with_adapter = PeftModel.from_pretrained(
        base, adapter_folder,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    proc = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", use_fast=True)
    size = with_adapter.num_parameters() / 1e9
    label = f"{size:.1f}B" if size >= 1 else f"{int(size*1e3)}M"
    print(f"Loaded model with {label} parameters")
    return with_adapter.eval(), proc

def run_inference(images_dir: str, csv_file: str, model, processor):
    df = pd.read_csv(csv_file)
    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(images_dir, row["image_name"])
        prompt = f"<image> Answer the question in exactly one word: {row['question']}"
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        offset = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=100, do_sample=False)[0][offset:]
        answer = processor.decode(gen, skip_special_tokens=True).split()[0].lower() or "error"
        outputs.append(answer)
    df["generated_answer"] = outputs
    df.to_csv("results.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--csv_path",   required=True)
    args = parser.parse_args()
    adapter_folder = fetch_adapter_from_onedrive(ONEDRIVE_URL, "adapter_data")
    print(f"Adapter folder: {adapter_folder}")
    model, proc = build_model(adapter_folder)
    run_inference(args.image_dir, args.csv_path, model, proc)

if __name__ == "__main__":
    main()
