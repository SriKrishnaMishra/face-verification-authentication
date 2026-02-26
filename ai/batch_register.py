import os
import argparse
from pathlib import Path
from ai.face_embedding_manager import extract_embedding, save_embedding


def process_folder(folder: str, model_name: str = 'VGG-Face', pattern: str = '*.*'):
    """Process all image files in folder. Filenames or subfolder names are used as user IDs.

    Rules:
    - If images are in subfolders, subfolder name is the user_id and all images inside are registered to that user.
    - Otherwise, filenames (without extension) are used as user_id.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    # Check subfolders
    for child in folder.iterdir():
        if child.is_dir():
            user_id = child.name
            for img in child.rglob(pattern):
                if img.is_file():
                    try:
                        emb = extract_embedding(str(img), model_name=model_name)
                        save_embedding(f"{user_id}", emb)
                        print(f"Saved embedding for {user_id} from {img}")
                    except Exception as e:
                        print(f"Failed {img}: {e}")
    # Also process files directly under folder
    for img in folder.glob(pattern):
        if img.is_file():
            user_id = img.stem
            try:
                emb = extract_embedding(str(img), model_name=model_name)
                save_embedding(user_id, emb)
                print(f"Saved embedding for {user_id} from {img}")
            except Exception as e:
                print(f"Failed {img}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch register images into embeddings.json')
    parser.add_argument('folder', help='Folder with images or subfolders per user')
    parser.add_argument('--model', default='VGG-Face', help='DeepFace model name')
    args = parser.parse_args()
    process_folder(args.folder, model_name=args.model)
