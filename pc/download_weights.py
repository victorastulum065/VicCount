"""
Descarcă O SINGURĂ DATĂ tot ce trebuie (≈1.7 GB). După asta VicCount merge offline.
  - BERT (bert-base-uncased) -> checkpoints/bert-base-uncased
  - CountGD++ (countgd_plusplus.pth, 1.25 GB) -> checkpoints/
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints")
BERT = os.path.join(CKPT, "bert-base-uncased")
WEIGHTS = os.path.join(CKPT, "countgd_plusplus.pth")
HF_REPO = "nikigoli/CountGDPlusPlus"
GDRIVE_ID = "1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs"  # link oficial din README CountGD++

os.makedirs(CKPT, exist_ok=True)
for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(k, None)


def get_bert():
    if os.path.isfile(os.path.join(BERT, "config.json")):
        print("✓ BERT deja există")
        return
    print("↓ BERT bert-base-uncased…")
    from transformers import AutoTokenizer, BertConfig, BertModel

    cfg = BertConfig.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=cfg).save_pretrained(BERT)
    cfg.save_pretrained(BERT)
    AutoTokenizer.from_pretrained("bert-base-uncased").save_pretrained(BERT)
    print("✓ BERT salvat")


def get_weights():
    if os.path.isfile(WEIGHTS) and os.path.getsize(WEIGHTS) > 500e6:
        print("✓ CountGD++ deja există")
        return
    # 1) Hugging Face
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files(HF_REPO) if f.endswith((".pth", ".pt", ".ckpt"))]
        if files:
            pick = next((f for f in files if "plus" in f.lower()), files[0])
            print(f"↓ CountGD++ din HuggingFace: {HF_REPO}/{pick}")
            p = hf_hub_download(HF_REPO, pick)
            shutil.copyfile(p, WEIGHTS)
            print("✓ CountGD++ salvat")
            return
    except Exception as e:
        print(f"  (HuggingFace indisponibil: {e.__class__.__name__}) — încerc Google Drive")
    # 2) Google Drive
    import gdown

    print("↓ CountGD++ din Google Drive (1.25 GB)…")
    gdown.download(id=GDRIVE_ID, output=WEIGHTS, quiet=False)
    if not os.path.isfile(WEIGHTS) or os.path.getsize(WEIGHTS) < 500e6:
        sys.exit(
            "✗ Descărcarea a eșuat. Descarcă manual de aici și pune fișierul în checkpoints/countgd_plusplus.pth:\n"
            f"  https://huggingface.co/{HF_REPO}\n"
            f"  https://drive.google.com/file/d/{GDRIVE_ID}/view"
        )
    print("✓ CountGD++ salvat")


if __name__ == "__main__":
    get_bert()
    get_weights()
    print("\nGata. De acum merge offline: python server.py")
