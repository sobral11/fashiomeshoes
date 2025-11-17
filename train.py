import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def parse_roboflow_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    available_splits = {}
    if os.path.exists(train_dir):
        available_splits['train'] = train_dir
    if os.path.exists(valid_dir):
        available_splits['valid'] = valid_dir
    if os.path.exists(test_dir):
        available_splits['test'] = test_dir

    if not available_splits:
        raise FileNotFoundError(f"No dataset folders found in: {data_dir}")

    print(f"Found folders: {list(available_splits.keys())}")

    all_images = {}
    category_images = {}
    categories_set = set()

    for split_name, split_dir in available_splits.items():
        coco_file = os.path.join(split_dir, '_annotations.coco.json')

        if not os.path.exists(coco_file):
            print(f"COCO file not found in {split_dir}, skipping...")
            continue

        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        images = {img['id']: img['file_name'] for img in coco_data['images']}

        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']

            if image_id in images and category_id in categories:
                filename = images[image_id]
                full_path = os.path.join(split_dir, filename)
                category_name = categories[category_id]

                if os.path.exists(full_path):
                    all_images[full_path] = category_name
                    categories_set.add(category_name)

                    if category_name not in category_images:
                        category_images[category_name] = []
                    category_images[category_name].append(full_path)

    if not all_images:
        raise ValueError("No valid images found in dataset!")

    categories_list = sorted(list(categories_set))

    print("=" * 60)
    print("ROBOFLOW DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total images: {len(all_images)}")
    print(f"Categories ({len(categories_list)}): {categories_list}")
    print("DETAILED STATISTICS:")
    for cat_name in sorted(category_images.keys()):
        print(f"  {cat_name}: {len(category_images[cat_name])} images")

    return categories_list, category_images, all_images


def create_grouped_dataset_split(category_images, group_size=800, val_size=0.15):
    groups = []

    print(f"CREATING GROUPS OF {group_size} IMAGES...")

    for category_name, images in category_images.items():
        print(f"Processing {category_name}: {len(images)} images")

        num_groups = max(1, len(images) // group_size)
        images_per_group = len(images) // num_groups

        for group_idx in range(num_groups):
            start_idx = group_idx * images_per_group
            end_idx = start_idx + images_per_group if group_idx < num_groups - 1 else len(images)

            group_images = images[start_idx:end_idx]

            if len(group_images) > 1:
                train_imgs, val_imgs = train_test_split(
                    group_images,
                    test_size=val_size,
                    random_state=42 + group_idx,
                    stratify=[category_name] * len(group_images)
                )
            else:
                train_imgs = group_images
                val_imgs = []

            groups.append({
                'category': category_name,
                'train_images': train_imgs,
                'val_images': val_imgs,
                'group_id': group_idx
            })

            print(f"  Group {group_idx}: {len(train_imgs)} train, {len(val_imgs)} val")

    return groups


def create_balanced_mapping_from_groups(groups, categories):
    train_mapping = {}
    val_mapping = {}

    images_per_category = {cat: 0 for cat in categories}
    val_per_category = {cat: 0 for cat in categories}

    for group in groups:
        category = group['category']

        for img_path in group['train_images']:
            train_mapping[img_path] = category
            images_per_category[category] += 1

        for img_path in group['val_images']:
            val_mapping[img_path] = category
            val_per_category[category] += 1

    print("FINAL GROUPED DATASET STATISTICS:")
    print(f"Total training images: {len(train_mapping)}")
    print(f"Total validation images: {len(val_mapping)}")

    for cat in categories:
        print(f"  {cat}: {images_per_category[cat]} train, {val_per_category[cat]} val")

    return train_mapping, val_mapping


def calculate_class_weights(mapping, category_to_id):
    labels = [category_to_id[cat] for cat in mapping.values()]
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    return dict(zip(unique_labels, class_weights))


class GroupedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, mapping, category_to_id, batch_size=16, target_size=(224, 224), augment=False):
        super().__init__()
        self.batch_size = batch_size
        self.target_size = target_size

        self.image_paths = []
        self.labels = []

        for img_path, category_name in mapping.items():
            self.image_paths.append(img_path)
            self.labels.append(category_to_id[category_name])

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        if augment:
            self.augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.02),
                layers.RandomZoom(0.05),
                layers.RandomContrast(0.1),
            ])
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        batch_paths = self.image_paths[start_idx:end_idx]
        batch_y = self.labels[start_idx:end_idx]

        batch_x = []
        for img_path in batch_paths:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array * 255.0)
                    batch_x.append(img_array)
            except Exception as e:
                batch_x.append(np.zeros((*self.target_size, 3), dtype=np.float32))

        batch_x = np.array(batch_x)

        if self.augmentation and np.random.random() > 0.5:
            batch_x = self.augmentation(batch_x, training=True)

        return batch_x, batch_y

    def on_epoch_end(self):
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = self.image_paths[indices]
        self.labels = self.labels[indices]


def build_model(num_classes):
    print(f"Building EfficientNetB0 model for {num_classes} categories...")

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def save_category_mapping(model_save_path, category_to_id, categories):
    mapping_file = os.path.join(model_save_path, 'shoe_categories.json')

    index_to_category = {idx: cat for cat, idx in category_to_id.items()}
    mapping_data = {
        'category_to_index': category_to_id,
        'index_to_category': index_to_category,
        'num_classes': len(category_to_id),
        'categories_list': categories
    }

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)

    print(f"Category file saved at: {mapping_file}")


def train_model(data_dir, batch_size=16, epochs=30, group_size=800):
    configure_gpu()

    print("STARTING GROUPED TRAINING APPROACH")
    print(f"Dataset: {data_dir}")
    print(f"Group size: {group_size}")

    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return None, 0

    try:
        categories, category_images, all_images = parse_roboflow_dataset(data_dir)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None, 0

    if not categories:
        print("ERROR: No categories found!")
        return None, 0

    groups = create_grouped_dataset_split(category_images, group_size=group_size)
    train_mapping, val_mapping = create_balanced_mapping_from_groups(groups, categories)

    if len(train_mapping) == 0:
        print("ERROR: No images found for training!")
        return None, 0

    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    num_classes = len(categories)

    class_weights = calculate_class_weights(train_mapping, category_to_id)
    print(f"Class weights: {class_weights}")

    models_dir = os.path.join(os.path.dirname(data_dir), 'grouped_shoe_models')
    os.makedirs(models_dir, exist_ok=True)

    save_category_mapping(models_dir, category_to_id, categories)

    train_gen = GroupedDataGenerator(
        train_mapping, category_to_id,
        batch_size=batch_size,
        target_size=(224, 224),
        augment=True
    )

    val_gen = GroupedDataGenerator(
        val_mapping, category_to_id,
        batch_size=batch_size,
        target_size=(224, 224),
        augment=False
    )

    print("Building model...")
    model = build_model(num_classes)

    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'shoe_model_best.h5'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1
        ),
        CSVLogger(os.path.join(models_dir, 'shoe_training_log.csv'))
    ]

    model.compile(
        optimizer=Adam(learning_rate=3e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])

    model.save(os.path.join(models_dir, 'shoe_model_final.h5'))

    print("=" * 60)
    print("GROUPED TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total training images: {len(train_mapping)}")
    print(f"Total validation images: {len(val_mapping)}")
    print(f"MODEL SAVED AT: {models_dir}")

    return history.history, best_val_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Grouped Transfer Learning for Shoes')
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--group_size', type=int, default=800, help='Images per group')

    args = parser.parse_args()

    history, best_val_acc = train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        group_size=args.group_size
    )


if __name__ == "__main__":
    main()