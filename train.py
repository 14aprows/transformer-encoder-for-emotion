import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config.config import Config
from data.build_vocab import Vocab as build_vocab
from data.dataset import EmotionDataset
from model.transformer_classifier import TransformerClassifier
from trainer.trainer import Trainer

def main():
    torch.manual_seed(42)
    config = Config()

    raw_train = load_dataset("emotion", split="train")
    texts = [item["text"] for item in raw_train]
    vocab = build_vocab(texts)
    config.vocab_size = len(vocab)

    train_ds = EmotionDataset(split="train", vocab=vocab, max_len=config.max_len)
    val_ds = EmotionDataset(split="validation", vocab=vocab, max_len=config.max_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = TransformerClassifier(
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_len=config.max_len,
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit(epochs=config.epochs)

if __name__ == "__main__":
    main()