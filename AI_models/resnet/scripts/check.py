import torch

train_images = torch.load("data/processed/train_images.pt")
print("Train images shape:", train_images.shape)

test_images = torch.load("data/processed/test_images.pt")
print("Test images shape:", test_images.shape)