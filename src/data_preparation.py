import os
import random
import shutil


def split_dataset(source_dir, output_dir):
    random.seed(42)

    classes = ["Cat", "Dog"]

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)

        images = [
            file_name
            for file_name in os.listdir(class_path)
            if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        total_images = len(images)

        train_end = int(total_images * 0.8)
        val_end = int(total_images * 0.9)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():
            destination_folder = os.path.join(
                output_dir,
                split_name,
                class_name
            )

            os.makedirs(destination_folder, exist_ok=True)

            for image_name in split_images:
                source_file = os.path.join(class_path, image_name)

                destination_file = os.path.join(
                    destination_folder,
                    image_name
                )

                shutil.copy2(source_file, destination_file)

        print(
            f"{class_name}: "
            f"{len(train_images)} train, "
            f"{len(val_images)} validation, "
            f"{len(test_images)} test"
        )


if __name__ == "__main__":
    split_dataset(
        source_dir="data/raw/PetImages",
        output_dir="data/processed"
    )