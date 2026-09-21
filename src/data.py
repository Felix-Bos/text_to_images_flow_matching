import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Variables 

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255), 
}
COLOR_NAMES = list(COLORS.keys())

DIGIT_CHARS_TO_WORDS = {
    '0': "zero",
    '1': "one",
    '2': "two",
    '3': "three",
    '4': "four",
    '5': "five",
    '6': "six",
    '7': "seven",
    '8': "eight",
    '9': "nine"
}

VOCAB = ["<pad>", "a", "rotated", "degrees"] + COLOR_NAMES + list(DIGIT_CHARS_TO_WORDS.values()) + list(DIGIT_CHARS_TO_WORDS.keys())
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# longueur max de séquence : "a" + couleur + digit_word + "rotated" + jusqu'à
# 3 chiffres (angle 0-359) + "degrees" = 4 + 3 + 1 = 8
MAX_SEQ_LENGTH = 8


def caption_to_tokens(color_name: str, digit: int, angle: int) -> tuple:
    """
    Tokenizes the input color name, digit, and angle into a list of token indices.
    """
    tokens = ["a", color_name, DIGIT_CHARS_TO_WORDS[str(digit)], "rotated"]
    
    # Split the angle into individual digits and convert to words
    angle_int = int(round(angle)) % 360  # Ensure angle is within 0-359
    words = ["a", color_name, DIGIT_CHARS_TO_WORDS[str(digit)], "rotated"] + [i for i in str(angle_int)] + ["degrees"]
    token_indices = [TOKEN_TO_IDX[token] for token in words]
    
    n = len(token_indices)
    if n < MAX_SEQ_LENGTH:
        token_indices += [TOKEN_TO_IDX["<pad>"]] * (MAX_SEQ_LENGTH - n)
    
    mask = [False] * n + [True] * (MAX_SEQ_LENGTH - n)
    
    return torch.tensor(token_indices, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def caption_to_string(color_name: str, digit: int, angle: int) -> str:
    angle_int = int(round(angle)) % 360
    return f"a {color_name} {DIGIT_CHARS_TO_WORDS[str(digit)]} rotated {angle_int} degrees"


class MNISTModified(Dataset):
    def __init__(self, root='./data', train=True, download=True, seed = 0, rotate_range=(0.0, 360)):
        self.mnist_data = datasets.MNIST(root=root, train=train, download=download)
        self.seed = seed
        self.rotate_range = rotate_range
        
        self.resize = transforms.Resize((32, 32))
        self.rng = random.Random(seed)
        
        n = len(self.mnist_data)
        self.indices = list(range(n))
        
        self.colors_per_items = [self.rng.choice(COLOR_NAMES) for _ in range(n)]
        low, high = rotate_range
        self.angles_per_items = [self.rng.uniform(low, high) for _ in range(n)]
        
        
    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        image = self.resize(image)
        
        angle = self.angles_per_items[idx]
        if angle != 0.0:
            image = transforms.functional.rotate(image, angle, fill=0)
        
        color_name = self.colors_per_items[idx]
        grayscale_image = transforms.functional.to_tensor(image)
        color = torch.tensor(COLORS[color_name], dtype=torch.float32) / 255
        color_image = grayscale_image.repeat(3, 1, 1) * color.view(3, 1, 1)
    
        tokens, padding_mask = caption_to_tokens(color_name, label, angle)
        caption = caption_to_string(color_name, label, angle)
        
        return color_image, tokens, padding_mask, caption


if __name__ == "__main__":
    dataset = MNISTModified(train=True, download=True, seed=42)
    print(f"Dataset length: {len(dataset)}")
    
    # Test the first item
    image, tokens, padding_mask, caption = dataset[0]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(caption)
    plt.axis('off')
    plt.show()
    
    print(f"Image shape: {image.shape}")
    print(f"Tokens: {tokens}")
    print(f"Padding mask: {padding_mask}")
    print(f"Caption: {caption}")