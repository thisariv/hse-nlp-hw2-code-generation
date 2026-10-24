"""Run using Kaggle CLI interpreter; the library owns authentication, no key inspection."""
import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();ref=sys.argv[1];page=1
# GetDataset returns403 for nonexistent private slugs too. Enumerate mine; never infer absence from403.
while True:
    rows=api.dataset_list(mine=True,page=page) or []
    found=next((d for d in rows if d.ref==ref),None)
    if found:
        print(json.dumps({'exists':True,'private':found.is_private,'version':found.current_version_number}));break
    if not rows:
        print(json.dumps({'exists':False}));break
    page+=1
