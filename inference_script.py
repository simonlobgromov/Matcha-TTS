from pathlib import Path
import argparse
import soundfile as sf
import torch
import io
import argparse
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse




def load_matcha( checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_vocoder(checkpoint_path, device):
    vocoder = None
    vocoder = load_hifigan(checkpoint_path, device)
    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser

def process_text(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["kyrgyz_cleaners"]), 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()

@torch.inference_mode()
def process_text_gradio(text):
    output = process_text(1, text, device)
    return output["x_phones"][1::2], output["x"], output["x_lengths"]

@torch.inference_mode()
def synthesise_mel(text, text_length, n_timesteps, temperature, length_scale, spk=-1):
    spk = torch.tensor([spk], device=device, dtype=torch.long) if spk >= 0 else None
    output = model.synthesise(
        text,
        text_length,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    sf.write('./output/out_audio.wav', output["waveform"], 22050, "PCM_24")

def get_inference(text, n_timesteps=20, mel_temp = 0.667, length_scale=0.8, spk=-1):
    phones, text, text_lengths = process_text_gradio(text)
    synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)


def tensor_to_wav_bytes(tensor_audio, sample_rate=22050): # Байтовый формат
    waveform = tensor_audio.cpu().numpy()
    with io.BytesIO() as buffer:
        sf.write(buffer, waveform, sample_rate, format='WAV')
        wav_bytes = buffer.getvalue()
    return wav_bytes



device = torch.device("cpu")
model_path = './checkpoints/checkpoint.ckpt'
vocoder_path = './checkpoints/generator'
model = load_matcha(model_path, device) 
vocoder, denoiser = load_vocoder(vocoder_path, device) 

def main():

    parser = argparse.ArgumentParser(
        description="Если возжелаете параметры которые вам угодны, Сэр))"
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=0.8,
        help="change the speaking rate, a higher value means slower speaking rate (default: 0.8)",
    )
    args = parser.parse_args()

    get_inference(text = args.text, length_scale=args.speaking_rate)




if __name__ == "__main__":
    main()
