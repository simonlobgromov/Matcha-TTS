<div align="center">



# 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching

### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Éva Székely](https://www.kth.se/profile/szekely), and [Gustav Eje Henter](https://people.kth.se/~ghe/)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p style="text-align: center;">
  <img src="https://shivammehta25.github.io/Matcha-TTS/images/logo.png" height="128"/>
</p>

</div>

> This is the official code implementation of 🍵 Matcha-TTS [ICASSP 2024].

We propose 🍵 Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses [conditional flow matching](https://arxiv.org/abs/2210.02747) (similar to [rectified flows](https://arxiv.org/abs/2209.03003)) to speed up ODE-based speech synthesis. Our method:

- Is probabilistic
- Has compact memory footprint
- Sounds highly natural
- Is very fast to synthesise from

Check out our [demo page](https://shivammehta25.github.io/Matcha-TTS) and read [our ICASSP 2024 paper](https://arxiv.org/abs/2309.03199) for more details.

[Pre-trained models](https://drive.google.com/drive/folders/17C_gYgEHOxI5ZypcfE_k1piKCtyR0isJ?usp=sharing) will be automatically downloaded with the CLI or gradio interface.

You can also [try 🍵 Matcha-TTS in your browser on HuggingFace 🤗 spaces](https://huggingface.co/spaces/shivammehta25/Matcha-TTS).

## Teaser video

[![Watch the video](https://img.youtube.com/vi/xmvJkz3bqw0/hqdefault.jpg)](https://youtu.be/xmvJkz3bqw0)

# Matcha-TTS for Kyrgyz language

## Train with Kany Dataset

The training dataset has 7016 samples and 13 hours of speech. All settings for training have already been made.

## Process by Terminal

* **Load this repo and connect to HF**

```
git clone https://github.com/simonlobgromov/Matcha-TTS
cd Matcha-TTS
pip install -e .
```
!!!The environment will be restarted!!!

Install this:

```
apt-get install espeak-ng
```
Connect to HF

```
git config --global credential.helper store
huggingface-cli login
```

* **Load the Data**

```
create-dataset

# If you see a cat, then everything is fine!
```

* **Train**

```
python matcha/train.py experiment=akylai
```

* **Checkpoints**

Checkpoints will be saved in `./Matcha-TTS/logs/train/akylai/runs/<DATE>_<TIME>/checkpoints`. Unload them or select the last few checkpoints.

