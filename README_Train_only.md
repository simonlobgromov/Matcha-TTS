<div align="center">



# AkylAI TTS


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p style="text-align: center;">
  <img src="https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png" height="400"/>
</p>

</div>



# AkylAI-TTS for Kyrgyz language

We present to you a model trained in the Kyrgyz language, which has been trained on 13 hours of speech and 7,000 samples, complete with source code and training scripts. The architecture is based on Matcha-TTS.
It`s a new approach to non-autoregressive neural TTS, that uses [conditional flow matching](https://arxiv.org/abs/2210.02747) (similar to [rectified flows](https://arxiv.org/abs/2209.03003)) to speed up ODE-based speech synthesis. Our method:

- Is probabilistic
- Has compact memory footprint
- Sounds highly natural
- Is very fast to synthesise from

You can try our *AkylAI TTS* by visiting [SPACE](https://huggingface.co/spaces/the-cramer-project/akylai-tts-mini) and read [ICASSP 2024 paper](https://arxiv.org/abs/2309.03199) for more details.

# Inference

## Run via terminal


It is recommended to start by setting up a virtual environment using `venv`.

1. Clone this repository and install all modules and dependencies by running the commands:

```
git clone https://github.com/simonlobgromov/Matcha-TTS
cd Matcha-TTS
pip install -e .
apt-get install espeak-ng
```


2. Run with CLI arguments:

- To synthesise from given text, run:

```bash
matcha-tts --text "<INPUT TEXT>"
```

- To synthesise from a file, run:

```bash
matcha-tts --file <PATH TO FILE>
```
- Speaking rate

```bash
matcha-tts --text "<INPUT TEXT>" --speaking_rate 1.0
```

- Sampling temperature

```bash
matcha-tts --text "<INPUT TEXT>" --temperature 0.667
```

- Euler ODE solver steps

```bash
matcha-tts --text "<INPUT TEXT>" --steps 10
```







# Train with your own dataset.

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

Checkpoints will be saved in `./Matcha-TTS/logs/train/<MODEL_NAME>/runs/<DATE>_<TIME>/checkpoints`. Unload them or select the last few checkpoints.

