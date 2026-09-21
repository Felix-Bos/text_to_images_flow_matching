import json
import os
import socket
import subprocess
import sys
import time
import webbrowser

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from src.data import VOCAB_SIZE, MAX_SEQ_LENGTH, MNISTModified, caption_to_tokens
from src.network import CONFIG_PATH, TextToImageModel
from src.flow_model import compute_loss, sample


PREVIEW_PROMPTS = [
    ("red", 3, 0),
    ("blue", 7, 90),
    ("green", 0, 180),
    ("purple", 5, 270),
]


def find_free_port(start_port=6006):
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("aucun port libre trouvé pour TensorBoard")


def launch_tensorboard(log_dir):
    """Lance `tensorboard --logdir <log_dir>` en arrière-plan et ouvre le navigateur.
    Retourne le Popen (ou None si tensorboard n'est pas installé) pour pouvoir
    l'arrêter proprement à la fin de l'entraînement."""
    port = find_free_port()
    url = f"http://localhost:{port}"
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("⚠️ tensorboard non trouvé, lancement automatique désactivé (pip install tensorboard)")
        return None

    print(f"TensorBoard lancé automatiquement : {url}")
    webbrowser.open(url)
    return process


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_preview_batch(device):
    pairs = [caption_to_tokens(color, digit, angle) for color, digit, angle in PREVIEW_PROMPTS]
    tokens = torch.stack([p[0] for p in pairs]).to(device)
    padding_mask = torch.stack([p[1] for p in pairs]).to(device)
    return tokens, padding_mask


@torch.no_grad()
def log_preview_images(model, writer, tokens, padding_mask, n_steps, img_size, img_channels, device, step):
    images = sample(
        model, tokens, padding_mask=padding_mask, n_steps=n_steps,
        img_shape=(img_channels, img_size, img_size), device=device,
    )
    grid = make_grid(images, nrow=len(PREVIEW_PROMPTS))
    writer.add_image("samples/preview", grid, global_step=step)
    model.train()


def train():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    training_cfg = config["training"]
    flow_cfg = config["flow_matching"]
    model_cfg = config["text_to_image_model"]
    general_cfg = config["general"]

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    lr = training_cfg["lr"]
    log_every = training_cfg["log_every"]
    sample_every = training_cfg["sample_every"]
    data_root = training_cfg["data_root"]
    ckpt_path = training_cfg["ckpt_path"]
    log_dir = training_cfg["log_dir"]
    path_type = flow_cfg["path_type"]
    n_steps = flow_cfg["n_steps"]
    img_size = general_cfg["img_size"]
    img_channels = general_cfg["img_channels"]

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    device = get_device()
    print(f"device: {device}")

    dataset = MNISTModified(root=data_root, train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = TextToImageModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LENGTH, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"modèle : {n_params / 1e6:.2f}M paramètres, {len(dataset)} images d'entraînement, path_type={path_type!r}")

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    tb_process = launch_tensorboard(log_dir)

    preview_tokens, preview_padding_mask = make_preview_batch(device)

    try:
        step = 0
        for epoch in range(epochs):
            running_loss = 0.0
            running_count = 0
            pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}", unit="batch")

            for images, tokens, padding_mask, _captions in pbar:
                images = images.to(device)
                tokens = tokens.to(device)
                padding_mask = padding_mask.to(device)

                loss = compute_loss(model, images, tokens, padding_mask=padding_mask, path_type=path_type)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value
                running_count += 1
                pbar.set_postfix(loss=f"{loss_value:.4f}")

                if step % log_every == 0:
                    writer.add_scalar("loss/train_step", loss_value, step)

                if step % sample_every == 0:
                    log_preview_images(
                        model, writer, preview_tokens, preview_padding_mask,
                        n_steps=n_steps, img_size=img_size, img_channels=img_channels,
                        device=device, step=step,
                    )

                step += 1

            epoch_loss = running_loss / max(running_count, 1)
            writer.add_scalar("loss/epoch", epoch_loss, epoch)
            writer.flush()

            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, ckpt_path)
            print(f"--- epoch {epoch + 1}/{epochs} terminée, loss moyenne {epoch_loss:.4f}, checkpoint sauvegardé dans {ckpt_path} ---")
    finally:
        writer.close()
        if tb_process is not None:
            print("arrêt de TensorBoard...")
            tb_process.terminate()

    return model


if __name__ == "__main__":
    train()
