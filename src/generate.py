import json
import re
import time

import gradio as gr
import numpy as np
import torch

from src.data import COLOR_NAMES, VOCAB_SIZE, MAX_SEQ_LENGTH, caption_to_tokens, caption_to_string
from src.network import CONFIG_PATH, TextToImageModel
from src.flow_model import sample_with_steps

DIGIT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path, device):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model_cfg = config["text_to_image_model"]
    flow_cfg = config["flow_matching"]
    general_cfg = config["general"]

    model = TextToImageModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LENGTH, **model_cfg).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        status = f"checkpoint chargé depuis {ckpt_path} (epoch {epoch})"
    except FileNotFoundError:
        status = f"⚠️ aucun checkpoint trouvé à {ckpt_path} -- poids aléatoires (non entraînés)"

    model.eval()
    return model, config, status


def parse_prompt(text):
    """Parse une phrase libre du style 'a red 7 rotated 45 degrees' -> (color, digit, angle).
    Accepte le chiffre en chiffre (7) ou en mot (seven). Lève ValueError si incomplet."""
    text = text.lower().strip()

    color = next((c for c in COLOR_NAMES if c in text), None)
    if color is None:
        raise ValueError(f"aucune couleur reconnue dans {text!r} (attendu parmi {COLOR_NAMES})")

    digit = None
    m = re.search(r"\b([0-9])\b", text)
    if m:
        digit = int(m.group(1))
    else:
        for word, value in DIGIT_WORDS.items():
            if word in text:
                digit = value
                break
    if digit is None:
        raise ValueError(f"aucun chiffre (0-9) reconnu dans {text!r}")

    angle_match = re.search(r"(-?\d+)\s*deg", text) or re.search(r"rotated\s+(-?\d+)", text)
    angle = int(angle_match.group(1)) if angle_match else 0

    return color, digit, angle


def tensor_to_numpy_image(img_tensor, upscale=16):
    # img_tensor : (C, H, W) dans [0, 1] -> (H, W, C) uint8, agrandi en nearest-neighbor
    # pour rester net (l'image source ne fait que 32x32) au lieu d'être flouté
    # par le redimensionnement bilinéaire du navigateur.
    img = img_tensor.clamp(0, 1)
    if upscale > 1:
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, scale_factor=upscale, mode="nearest")
        img = img.squeeze(0)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr


def build_app(ckpt_path="checkpoints/model.pt"):
    device = get_device()
    model, config, status = load_model(ckpt_path, device)
    flow_cfg = config["flow_matching"]
    general_cfg = config["general"]
    img_size = general_cfg["img_size"]
    img_channels = general_cfg["img_channels"]
    print(status)

    def generate(prompt_text, color_dd, digit_dd, angle_slider, n_steps, use_free_text):
        if use_free_text and prompt_text.strip():
            try:
                color, digit, angle = parse_prompt(prompt_text)
            except ValueError as e:
                yield None, f"Erreur de parsing : {e}"
                return
        else:
            color, digit, angle = color_dd, int(digit_dd), int(angle_slider)

        caption = caption_to_string(color, digit, angle)
        tokens, padding_mask = caption_to_tokens(color, digit, angle)
        tokens = tokens.unsqueeze(0).to(device)
        padding_mask = padding_mask.unsqueeze(0).to(device)

        start = time.time()
        n_steps = int(n_steps)
        for i, images in enumerate(sample_with_steps(
            model, tokens, padding_mask=padding_mask, n_steps=n_steps,
            img_shape=(img_channels, img_size, img_size), device=device,
        )):
            img_np = tensor_to_numpy_image(images[0])
            progress_msg = f"prompt: “{caption}” — étape {i + 1}/{n_steps}"
            yield img_np, progress_msg

        elapsed = time.time() - start
        yield img_np, f"prompt: “{caption}” — terminé en {elapsed:.2f}s ({n_steps} pas)"

    with gr.Blocks(title="Text-to-Image (Flow Matching)") as demo:
        gr.Markdown("# Générateur Text-to-Image (Flow Matching)")
        gr.Markdown(f"`{status}` — device: `{device}`")

        with gr.Row():
            with gr.Column():
                use_free_text = gr.Checkbox(label="Utiliser la phrase libre plutôt que les menus", value=True)
                prompt_text = gr.Textbox(
                    label="Phrase libre",
                    placeholder="a red 7 rotated 45 degrees",
                    value="a red 7 rotated 45 degrees",
                )
                gr.Markdown("*(ou utilise les menus ci-dessous en décochant la case)*")
                color_dd = gr.Dropdown(choices=COLOR_NAMES, value=COLOR_NAMES[0], label="Couleur")
                digit_dd = gr.Dropdown(choices=[str(d) for d in range(10)], value="7", label="Chiffre")
                angle_slider = gr.Slider(0, 359, value=0, step=1, label="Angle de rotation (degrés)")
                n_steps = gr.Slider(5, 200, value=flow_cfg["n_steps"], step=1, label="Nombre de pas d'intégration")
                btn = gr.Button("Générer", variant="primary")

            with gr.Column():
                image_out = gr.Image(
                    label="Image générée (mise à jour en temps réel)", type="numpy",
                    image_mode="RGB", elem_id="generated-image",
                    sources=[], interactive=False,
                )
                status_out = gr.Textbox(label="Statut", interactive=False)

        btn.click(
            fn=generate,
            inputs=[prompt_text, color_dd, digit_dd, angle_slider, n_steps, use_free_text],
            outputs=[image_out, status_out],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    args = parser.parse_args()

    css = """
    #generated-image { min-height: 512px; }
    #generated-image img { object-fit: contain; image-rendering: pixelated; }
    """

    app = build_app(ckpt_path=args.ckpt)
    app.queue().launch(css=css)
