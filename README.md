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

