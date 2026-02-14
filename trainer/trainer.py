import torch 
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import accuracy

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.scaler = GradScaler()
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_acc = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, leave=True)
        for batch in loop:
            x = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=(self.device == "cuda"), device_type=self.device):
                pred = self.model(x, mask)
                loss = self.criterion(pred, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()

            loop.set_postfix(loss=total_loss / len(self.train_loader))
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_acc = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["label"].to(self.device)

                out = self.model(x, mask)
                total_acc += accuracy(out, y)

        return total_acc / len(self.val_loader)
    
    def fit(self, epochs=10):
        for epoch in range(1, epochs+1):
            train_loss = self.train_epoch()
            val_acc = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch}, Train Loss {train_loss:.4f}, Val Acc {val_acc:.2f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_checkpoint(self.model, self.optimizer, epoch, os.path.join(self.checkpoint_dir, "best.pth"))
                print("Saved best model")