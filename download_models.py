from pathlib import Path

from app.helpers.downloader import download_file
from app.processors.models_data import models_list

for model_data in models_list:
    model_name = model_data['model_name']
    local_path = Path(model_data['local_path'])
    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = model_data.get('url')
    hash_value = model_data.get('hash')

    if not url:
        manual_url = model_data.get('manual_url')
        manual_note = model_data.get('manual_note')

        if local_path.exists():
            print(f"\nSkipping {model_name}; checkpoint already present at {local_path}.")
            continue

        print(f"\n{model_name} must be downloaded manually.")
        if manual_note:
            print(manual_note)
        if manual_url:
            print(f"Download URL: {manual_url}")
        print(f"Place the file at: {local_path}")
        continue

    download_file(model_name, str(local_path), hash_value, url)
