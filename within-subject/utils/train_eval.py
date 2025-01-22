# utils/train_eval.py
# -*- coding: utf-8 -*-

"""
Training and Evaluation utilities (Optimized for clarity and minimal console output).

Functionality:
1. train_epoch: Train the model for exactly 1 epoch with tqdm progress bar, 
   returning average training loss/accuracy.
2. validate_epoch: Evaluate model on validation set, returning loss/accuracy.
3. fit_and_evaluate: High-level training loop that trains for num_epochs, 
   logs metrics, and does final evaluation & confusion matrix.
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """
    Train the model for 1 epoch (with progress bar), returning average train loss & accuracy.

    Args:
        model (nn.Module): Model to be trained.
        device (torch.device): Device (CPU or GPU).
        train_loader (DataLoader): DataLoader for training set.
        optimizer (torch.optim.Optimizer): Optimizer instance.

    Returns:
        (avg_loss, avg_accuracy): float, float
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch in pbar:
        data = batch['seq'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = outputs.argmax(dim=1, keepdim=True)
            correct = preds.eq(targets.view_as(preds)).sum().item()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def validate_epoch(
    model: nn.Module,
    device: torch.device,
    val_loader: DataLoader
) -> tuple[float, float]:
    """
    Evaluate the model on validation set for 1 epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device (CPU or GPU).
        val_loader (DataLoader): Validation data loader.

    Returns:
        (val_loss, val_accuracy): float, float
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            data = batch['seq'].to(device)
            targets = batch['label'].to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1, keepdim=True)
            correct = preds.eq(targets.view_as(preds)).sum().item()

            batch_size = len(targets)
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def final_evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    class_names: list[str]
):
    """
    Final evaluation on test/val set, prints classification report & confusion matrix.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device (CPU or GPU).
        test_loader (DataLoader): Data for final evaluation (val or test).
        class_names (list[str]): Class label names.
    """
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch['seq'].to(device)
            targets = batch['label'].to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            preds_list.extend(preds.cpu().tolist())
            trues_list.extend(targets.cpu().tolist())

    # Print classification report
    print("\n=== Classification Report ===\n")
    print(classification_report(trues_list, preds_list, target_names=class_names))


def fit_and_evaluate(
    model_cls,
    train_dir: str,
    val_dir: str,
    seq_length: int,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    transform,
    num_channels: int,
    num_classes: int,
    lr: float = 1e-3,  
):
    """
    High-level training loop:
    1) Instantiates model_cls -> model.
    2) Creates train/val dataloaders from the given directories.
    3) Trains for num_epochs, logging train/val metrics each epoch.
    4) Prints final results & classification report on the val set.

    Args:
        model_cls (type): Class or constructor returning an nn.Module instance.
        train_dir (str): Path to the training data.
        val_dir (str): Path to the validation data.
        seq_length (int): Sequence length for cropping.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size.
        device (torch.device): "cpu" or "cuda".
        transform (callable): Transform function for data augmentation.
        num_channels (int): Number of EEG channels.
        num_classes (int): Number of output classes.
        lr (float): Learning rate for optimizer.

    Returns:
        model (nn.Module): The trained model.
        history (dict): Contains the epoch-wise training history for plotting or analysis.
                        Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    from .dataset import SeqDataset

    print("Initializing model...")
    model = model_cls(n_chans=num_channels, n_times=seq_length, n_classes=num_classes)
    model.to(device)

    # Build dataset/dataloaders
    train_dataset = SeqDataset(
        root=train_dir,
        seq_length=seq_length,
        is_train=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataset = SeqDataset(
        root=val_dir,
        seq_length=seq_length,
        is_train=False,
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For storing epoch-wise metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer)
        val_loss, val_acc = validate_epoch(model, device, val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f"=== Batch {epoch // 10} ===")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

    print(f"\n-----> Best Val Accuracy at {best_epoch} Epoch: {best_val_acc:.2f}%.")

    final_evaluate(
        model=model,
        device=device,
        test_loader=val_loader,
        class_names=val_dataset.class_names
    )

    plot_history(history)
    plt.show()

    return model, history, best_val_acc


def plot_history(history: dict):
    """
    A utility function to plot training history curves. 
    This can be placed in a separate file or used inline in notebooks.

    Args:
        history (dict): Must contain 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))

    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CrossEntropyLoss')
    axes[0].set_title('Train and Valid Loss')
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train and Valid Accuracy')
    axes[1].legend()

    plt.show()