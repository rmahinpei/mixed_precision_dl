import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from lightning import Fabric
import torchmetrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from watermark import watermark

torch.manual_seed(123)

# Configuration
MODEL_NAME = "roberta-base"
DATASET_NAME = "emotion" # "imdb" or "emotion"
DATASET_CONFIGS = {
    "imdb": {
        "path": "imdb",
        "text_column": "text",
        "label_column": "label",
        "num_classes": 2,
        "has_validation": False,
    },
    "emotion": {
        "path": "emotion",
        "text_column": "text",
        "label_column": "label",
        "num_classes": 6,
        "has_validation": True,
    }
}

# Training settings
PRECISION = "16-mixed" 
BATCH_SIZE = 12
NUM_EPOCHS = 1
MAX_LENGTH = 512


# Learning rate grid search
LEARNING_RATES = [1e-5, 2e-5, 5e-5, 1e-4]


# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


# Load dataset configuration
config = DATASET_CONFIGS[DATASET_NAME]
NUM_CLASSES = config["num_classes"]
text_col = config["text_column"]
label_col = config["label_column"]

print(f"\nLoading {DATASET_NAME} dataset...")

# Load dataset based on configuration
if "subset" in config:
    dataset = load_dataset(config["path"], config["subset"])
else:
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


# Tokenization
print(f"\nLoading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

def tokenize_text(batch):
    return tokenizer(
        batch[text_col],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_text, batched=True, batch_size=None)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", label_col])
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Create dataloaders
if label_col != "label":
    tokenized_dataset = tokenized_dataset.rename_column(label_col, "label")

train_dataset = TextDataset(tokenized_dataset, partition_key="train")
val_dataset = TextDataset(tokenized_dataset, partition_key="validation")
test_dataset = TextDataset(tokenized_dataset, partition_key="test")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    drop_last=True,
)


# Training function
def train(num_epochs, model, optimizer, train_loader, val_loader, fabric, num_classes, verbose=True):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )
            optimizer.zero_grad()
            fabric.backward(outputs["loss"])
            optimizer.step()

            if verbose and not batch_idx % 300:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {outputs['loss']:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(outputs["logits"], 1)
                train_acc.update(predicted_labels, batch["label"])
            model.train()

        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)
            for batch in val_loader:
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"]
                )
                predicted_labels = torch.argmax(outputs["logits"], 1)
                val_acc.update(predicted_labels, batch["label"])

            val_acc_value = val_acc.compute()
            if verbose:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc_value*100:.2f}%")

            # Track best validation accuracy
            if val_acc_value > best_val_acc:
                best_val_acc = val_acc_value

            train_acc.reset(), val_acc.reset()

    return best_val_acc


# Initialize Fabric
print(f"\nInitializing Fabric with precision: {PRECISION}")
fabric = Fabric(accelerator="cuda", devices=1, precision=PRECISION)
fabric.launch()

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
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
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


# Final training with best learning rate
print("\n" + "="*60)
print("FINAL TRAINING WITH BEST LR")
print("="*60)
print(f"Using LR = {best_lr}")

torch.manual_seed(123) 
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
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

    for batch in test_loader:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        predicted_labels = torch.argmax(outputs["logits"], 1)
        test_acc.update(predicted_labels, batch["label"])

print(f"\nMemory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
print(f"Test accuracy: {test_acc.compute()*100:.2f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Best LR: {best_lr}")
print(f"Best Val Acc: {best_val_acc*100:.2f}%")
print(f"Final Test Acc: {test_acc.compute()*100:.2f}%")