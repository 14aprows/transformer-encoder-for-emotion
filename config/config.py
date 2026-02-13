class Config:
    vocab_size = 20000
    max_len = 128

    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 4
    dropout = 0.1

    num_classes = 6

    batch_size = 32
    lr = 3e-4
    epochs = 10

    device = "cuda"

    checkpoint_dir = "checkpoints"