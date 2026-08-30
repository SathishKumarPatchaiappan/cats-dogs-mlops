import os
from PIL import Image


def clean_images(folder):
    fixed = 0
    removed = 0

    for class_name in ["Cat", "Dog"]:
        class_folder = os.path.join(folder, class_name)

        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)

            try:
                with Image.open(file_path) as image:
                    image = image.convert("RGB")
                    image.save(file_path)
                    fixed += 1

            except Exception:
                print(f"Removing invalid image: {file_name}")
                os.remove(file_path)
                removed += 1

    print(f"Images checked: {fixed}")
    print(f"Invalid images removed: {removed}")


if __name__ == "__main__":
    clean_images("data/raw/PetImages")