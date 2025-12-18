import time
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from lightning import Fabric
import torchmetrics
from transformers import AutoImageProcessor, AutoModelForImageClassification
from watermark import watermark
from PIL import Image

torch.manual_seed(123)


# Configuration
MODEL_NAME = "microsoft/resnet-50"
DATASET_NAME = "fashion_mnist" # "cifar100" or "fashion_mnist"
DATASET_CONFIGS = {
    "cifar100": {
        "path": "cifar100",
        "num_classes": 100,
        "has_validation": False,
        "image_key": "img",
        "label_key": "fine_label",
    },
    "fashion_mnist": {
        "path": "fashion_mnist",
        "num_classes": 10,
        "has_validation": False,
        "image_key": "image",
        "label_key": "label",
    }
}


# Training settings
PRECISION = "16-true"
BATCH_SIZE = 64
NUM_EPOCHS = 1


# Learning rate grid search
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4]


# Dataset class
class VisionDataset(Dataset):
    def __init__(self, dataset_dict, partition_key, image_processor, image_key, label_key):
        self.partition = dataset_dict[partition_key]
        self.image_processor = image_processor
        self.image_key = image_key
        self.label_key = label_key

    def __getitem__(self, index):
        item = self.partition[index]
        image = item[self.image_key]

        if image.mode != "RGB":
            image = image.convert("RGB")

        processed = self.image_processor(image, return_tensors="pt")

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "label": item[self.label_key]
        }

    def __len__(self):
        return self.partition.num_rows


# Load dataset configuration
config = DATASET_CONFIGS[DATASET_NAME]
NUM_CLASSES = config["num_classes"]
image_key = config["image_key"]
label_key = config["label_key"]

print(f"\nLoading {DATASET_NAME} dataset...")
dataset = load_dataset(config["path"])

# Create validation split if needed
if not config["has_validation"]:
    print("Creating train/validation split...")
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=123)
    dataset["train"] = train_val["train"]
    dataset["validation"] = train_val["test"]

print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")


# Load image processor
print(f"\nLoading image processor for {MODEL_NAME}...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
print(f"Image processor loaded successfully")


# Create dataloaders
train_dataset = VisionDataset(dataset, "train", image_processor, image_key, label_key)
val_dataset = VisionDataset(dataset, "validation", image_processor, image_key, label_key)
test_dataset = VisionDataset(dataset, "test", image_processor, image_key, label_key)

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)


# Training function
def train(num_epochs, model, optimizer, train_loader, val_loader, fabric, num_classes, verbose=True):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)

        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            outputs = model(
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )

            optimizer.zero_grad()
            fabric.backward(outputs.loss)
            optimizer.step()

            total_loss += outputs.loss.item()

            if verbose and batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {outputs.loss:.4f}")

            with torch.no_grad():
                predicted_labels = torch.argmax(outputs.logits, 1)
                train_acc.update(predicted_labels, batch["labels"])

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)
            val_loss = 0
            for batch in val_loader:
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"]
                )
                predicted_labels = torch.argmax(outputs.logits, 1)
                val_acc.update(predicted_labels, batch["labels"])
                val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc_value = val_acc.compute()

            if verbose:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc.compute()*100:.2f}% | Val Acc: {val_acc_value*100:.2f}%")

            if val_acc_value > best_val_acc:
                best_val_acc = val_acc_value

            train_acc.reset(), val_acc.reset()

    return best_val_acc


# Initialize Fabric
print(f"\nInitializing Fabric with precision: {PRECISION}")
fabric = Fabric(accelerator="cuda", devices=1, precision=PRECISION)
fabric.launch()

# Setup dataloaders once
train_loader, val_loader, test_loader = fabric.setup_dataloaders(
    train_loader, val_loader, test_loader
)


# Learning rate grid search
print("\n" + "="*60)
print("LEARNING RATE GRID SEARCH")
print("="*60)

best_lr = None
best_val_acc = 0.0
lr_results = {}

for lr in LEARNING_RATES:
    print(f"\n>>> Testing LR = {lr}")
    torch.manual_seed(123)  
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True  
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = fabric.setup(model, optimizer)

    start = time.time()
    val_acc = train(
        num_epochs=NUM_EPOCHS,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        fabric=fabric,
        num_classes=NUM_CLASSES,
        verbose=False  
    )
    elapsed = time.time() - start

    lr_results[lr] = val_acc.item()
    print(f"    Val Accuracy: {val_acc*100:.2f}% | Time: {elapsed/60:.2f} min")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

    del model, optimizer
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("GRID SEARCH RESULTS")
print("="*60)
for lr, acc in lr_results.items():
    marker = " <-- BEST" if lr == best_lr else ""
    print(f"LR {lr}: {acc*100:.2f}%{marker}")
print(f"\nBest LR: {best_lr} with Val Acc: {best_val_acc*100:.2f}%")


# Final training with best LR
print("\n" + "="*60)
print("FINAL TRAINING WITH BEST LR")
print("="*60)
print(f"Using LR = {best_lr}")

torch.manual_seed(123)  
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
model, optimizer = fabric.setup(model, optimizer)

start = time.time()
train(
    num_epochs=NUM_EPOCHS,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    fabric=fabric,
    num_classes=NUM_CLASSES,
    verbose=True
)
elapsed = time.time() - start
print(f"\nFinal training time: {elapsed/60:.2f} min")


# Test evaluation
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

with torch.no_grad():
    model.eval()
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(fabric.device)
    test_loss = 0

    for batch in test_loader:
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        predicted_labels = torch.argmax(outputs.logits, 1)
        test_acc.update(predicted_labels, batch["labels"])
        test_loss += outputs.loss.item()

    avg_test_loss = test_loss / len(test_loader)

print(f"\nMemory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
print(f"Test loss: {avg_test_loss:.4f}")
print(f"Test accuracy: {test_acc.compute()*100:.2f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Best LR: {best_lr}")
print(f"Best Val Acc: {best_val_acc*100:.2f}%")
print(f"Final Test Acc: {test_acc.compute()*100:.2f}%")