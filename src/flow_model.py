import json
import math

import torch
import torch.nn.functional as F

from src.network import CONFIG_PATH, TextToImageModel

PATH_TYPES = ("linear", "cosine")


def load_flow_matching_config(config_path=CONFIG_PATH):
    with open(config_path) as f:
        config = json.load(f)
    return config["flow_matching"]


def _path_xt_and_target(path_type, x0, x1, t_broadcast):
    """
    Calcule x_t et la vitesse cible target_v = dx_t/dt pour le chemin choisi.

    linear : x_t = (1-t)*x0 + t*x1                 (rectified flow, vitesse constante)
    cosine : x_t = cos(t*pi/2)*x0 + sin(t*pi/2)*x1  (interpolation trigonométrique)
    """
    if path_type == "linear":
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        target_v = x1 - x0
    elif path_type == "cosine":
        angle = t_broadcast * (math.pi / 2)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        x_t = cos_a * x0 + sin_a * x1
        target_v = (math.pi / 2) * (cos_a * x1 - sin_a * x0)
    else:
        raise ValueError(f"path_type inconnu: {path_type!r}, attendu parmi {PATH_TYPES}")

    return x_t, target_v


def compute_loss(model, x1, tokens, padding_mask=None, path_type="linear"):
    """
    x1           : (B, C, H, W) -- images réelles du batch, dans [-1, 1]
    tokens       : (B, L) long -- légendes tokenisées
    padding_mask : (B, L) bool ou None
    path_type    : "linear" (constant/rectified) ou "cosine" (sinusoidal)

    Retour : un scalaire (la loss), prêt pour .backward()
    """
    B = x1.shape[0]
    device = x1.device

    x0 = torch.randn_like(x1)                      # bruit gaussien, même shape que x1
    t = torch.rand(B, device=device)                # un t différent par exemple du batch, dans [0,1]

    t_broadcast = t[:, None, None, None]            # (B,1,1,1) pour multiplier avec (B,C,H,W)
    x_t, target_v = _path_xt_and_target(path_type, x0, x1, t_broadcast)

    v_pred = model(x_t, t, tokens, padding_mask=padding_mask)  # t reste (B,) ici, pas broadcasté

    return F.mse_loss(v_pred, target_v)


@torch.no_grad()
def sample(model, tokens, padding_mask=None, n_steps=50, img_shape=(3, 32, 32), device="cpu"):
    """
    tokens       : (B, L) long -- les prompts pour lesquels générer une image
    padding_mask : (B, L) bool ou None
    n_steps      : nombre de pas d'intégration (plus = plus précis mais plus lent)
    img_shape    : (C, H, W) de l'image à générer

    Retour : x, shape (B, *img_shape), valeurs dans [0, 1]
    """
    x = None
    for x in sample_with_steps(model, tokens, padding_mask=padding_mask, n_steps=n_steps,
                                img_shape=img_shape, device=device):
        pass
    return x


@torch.no_grad()
def sample_with_steps(model, tokens, padding_mask=None, n_steps=50, img_shape=(3, 32, 32), device="cpu"):
    """
    Comme `sample`, mais un générateur qui yield l'image (dans [0, 1]) après
    chaque pas d'intégration -- pratique pour visualiser le débruitage en
    temps réel (Gradio, TensorBoard, etc). Le dernier élément yield est le
    résultat final, identique à ce que renverrait `sample`.
    """
    model.eval()
    B = tokens.shape[0]
    x = torch.randn(B, *img_shape, device=device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v_pred = model(x, t, tokens, padding_mask=padding_mask)
        x = x + v_pred * dt
        yield ((x.clamp(-1, 1) + 1) / 2)


if __name__ == "__main__":
    from src.data import VOCAB_SIZE, MAX_SEQ_LENGTH, caption_to_tokens

    torch.manual_seed(0)
    B = 4
    model = TextToImageModel(
        vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LENGTH,
        embedding_dim=64, time_emb_dim=32, base_channels=16,
    )

    flow_cfg = load_flow_matching_config()
    default_path_type = flow_cfg["path_type"]
    default_n_steps = flow_cfg["n_steps"]
    print(f"config.json [flow_matching] -> path_type={default_path_type!r}, n_steps={default_n_steps}")

    # --- test de compute_loss (pour chaque path_type) ---
    x1 = torch.rand(B, 3, 32, 32) * 2 - 1  # simule des images réelles dans [-1,1]
    pairs = [caption_to_tokens("red", 7, 90) for _ in range(B)]
    tokens = torch.stack([p[0] for p in pairs])
    mask = torch.stack([p[1] for p in pairs])

    for path_type in PATH_TYPES:
        loss = compute_loss(model, x1, tokens, padding_mask=mask, path_type=path_type)
        assert loss.dim() == 0, f"compute_loss doit retourner un scalaire, pas {loss.shape}"
        print(f"compute_loss[{path_type}] OK, loss = {loss.item():.4f}")

    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.grad is None)
    print(f"backward OK, params sans gradient : {n_none}")

    try:
        compute_loss(model, x1, tokens, padding_mask=mask, path_type="bogus")
        raise AssertionError("compute_loss aurait dû lever ValueError pour un path_type inconnu")
    except ValueError:
        print("path_type invalide correctement rejeté")

    # --- test de sample (n_steps réduit pour le test, path_type depuis config.json) ---
    images = sample(model, tokens, padding_mask=mask, n_steps=5, img_shape=(3, 32, 32))
    expected_shape = (B, 3, 32, 32)
    assert tuple(images.shape) == expected_shape, f"sample: expected {expected_shape}, got {images.shape}"
    assert images.min() >= 0 and images.max() <= 1, "sample: les valeurs doivent être dans [0,1]"
    print(f"sample OK: {images.shape}, min={images.min().item():.3f}, max={images.max().item():.3f}")

    print("\nAll checks passed.")