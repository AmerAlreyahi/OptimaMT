"""Uploads the folders built by build_hf_repos.py to Hugging Face: one model
repo per pair, then (optionally) a Collection of them and the organization card.

    python scripts/upload_hf.py --dry-run                 # list what would be uploaded
    python scripts/upload_hf.py --private en-ar           # one pair, private
    python scripts/upload_hf.py --all --private           # every pair, private
    python scripts/upload_hf.py --all --collection        # every pair + the Collection
    python scripts/upload_hf.py --org-card                # the organization page (work/hf/_org_card)
    python scripts/upload_hf.py --space                   # the results page, a private static Space

The token comes from the HF_TOKEN environment variable, or from a `.env` file
next to project.toml (one line `HF_TOKEN=...`; .env is ignored by git). It is
never printed. Repo ids and the namespace come from project.toml.
"""

import argparse
import os
import sys

from common import CFG, HF, ROOT, repo_id, repo_name, results_url, sources, tagline

SKIP = ["*.part", ".DS_Store"]


def token() -> str:
    value = os.environ.get("HF_TOKEN")
    env = ROOT / ".env"
    if not value and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not value:
        raise SystemExit("Set HF_TOKEN (environment or .env next to project.toml).")
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", help="pairs such as en-ar")
    ap.add_argument("--all", action="store_true", help="every pair")
    ap.add_argument("--private", action="store_true", help="create the repos as private")
    ap.add_argument("--collection", action="store_true", help="create or update the Collection")
    ap.add_argument("--org-card", action="store_true", help="upload work/hf/_org_card/README.md as the org page")
    ap.add_argument("--space", action="store_true", help="upload work/hf/_space as a private static Space")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pairs = [e["pair"] for e in sources()] if a.all else a.pairs
    if not (pairs or a.collection or a.org_card or a.space):
        raise SystemExit("name the pairs (or --all), --collection, --org-card or --space")
    missing = [p for p in pairs if not (HF / repo_name(p) / "README.md").exists()]
    if missing:
        raise SystemExit(f"not built (run build_hf_repos.py): {' '.join(missing)}")
    if a.dry_run:
        for p in pairs:
            files = sorted(f.relative_to(HF / repo_name(p)).as_posix()
                           for f in (HF / repo_name(p)).rglob("*") if f.is_file())
            print(f"{repo_id(p)}  {'private' if a.private else 'public'}  {len(files)} files")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token())
    for p in pairs:
        rid = repo_id(p)
        api.create_repo(rid, repo_type="model", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(HF / repo_name(p)), ignore_patterns=SKIP,
                          commit_message=f"Add {CFG['name']} {p}")
        print(f"uploaded https://huggingface.co/{rid}", flush=True)

    if a.collection:
        title = f"{CFG['name']}: lighter OPUS-MT ONNX models"
        ns = CFG["hf_namespace"]
        existing = [c for c in api.list_collections(owner=ns) if c.title == title]
        col = existing[0] if existing else api.create_collection(
            title=title, namespace=ns, private=a.private,
            description=f"{CFG['name']}: {tagline()}"[:150])
        have = {i.item_id for i in api.get_collection(col.slug).items}
        for p in pairs or [e["pair"] for e in sources()]:
            if repo_id(p) not in have:
                api.add_collection_item(col.slug, item_id=repo_id(p), item_type="model", exists_ok=True)
        print(f"collection https://huggingface.co/collections/{col.slug}")

    if a.org_card:
        rid = f"{CFG['hf_namespace']}/README"
        api.create_repo(rid, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="space", folder_path=str(HF / "_org_card"),
                          delete_patterns=["style.css"], commit_message=f"{CFG['name']} organization card")
        print(f"organization card https://huggingface.co/{CFG['hf_namespace']}")

    if a.space:
        rid = f"{CFG['hf_namespace']}/{CFG['results_space']}"
        api.create_repo(rid, repo_type="space", space_sdk="static", private=True, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="space", folder_path=str(HF / "_space"), ignore_patterns=SKIP,
                          commit_message=f"{CFG['name']} results page")
        print(f"results page {results_url()} (private)")


if __name__ == "__main__":
    sys.exit(main())
