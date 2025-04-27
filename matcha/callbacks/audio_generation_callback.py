import os
import torch
from lightning.pytorch.callbacks import Callback
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse, plot_tensor, get_user_data_dir
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.hifigan.env import AttrDict
from matcha.hifigan.config import v1
from matcha.cli import VOCODER_URLS
import wget

class AudioGenerationCallback(Callback):
    def __init__(self, test_texts, vocoder_name="hifigan_T2_v1", every_n_epochs=10, enabled=True):
        super().__init__()
        self.test_texts = test_texts
        self.every_n_epochs = every_n_epochs
        self.vocoder_name = vocoder_name
        self.enabled = enabled
        self.vocoder = None
        
    def setup(self, trainer, pl_module, stage=None):
        if not self.enabled:
            return
            
        # Загрузка вокодера при первом запуске
        if self.vocoder is None:
            device = pl_module.device
            
            # Путь к вокодеру
            location = os.path.join(get_user_data_dir(), f"{self.vocoder_name}")
            
            # Проверка и загрузка вокодера, если его нет
            if not os.path.exists(location):
                os.makedirs(os.path.dirname(location), exist_ok=True)
                url = VOCODER_URLS.get(self.vocoder_name)
                if url:
                    print(f"Downloading vocoder from {url} to {location}")
                    wget.download(url=url, out=location)
                else:
                    raise ValueError(f"Vocoder {self.vocoder_name} not found in VOCODER_URLS")
            
            # Инициализация вокодера
            h = AttrDict(v1)
            self.vocoder = HiFiGAN(h).to(device)
            self.vocoder.load_state_dict(torch.load(location, map_location=device)["generator"])
            _ = self.vocoder.eval()
            self.vocoder.remove_weight_norm()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
            
        if trainer.current_epoch % self.every_n_epochs == 0:
            if not trainer.is_global_zero:
                return
                
            for i, text in enumerate(self.test_texts):
                # Преобразование текста в фонемы
                x = torch.tensor(
                    intersperse(text_to_sequence(text, ["kyrgyz_cleaners"]), 0),
                    dtype=torch.long,
                    device=pl_module.device,
                )[None]
                x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=pl_module.device)
                
                # Генерация мел-спектрограммы
                output = pl_module.synthesise(x, x_lengths, n_timesteps=20)
                
                # Логгирование мел-спектрограммы
                trainer.logger.experiment.add_image(
                    f"test_text_mel/{i}",
                    plot_tensor(output["decoder_outputs"].squeeze().cpu()),
                    trainer.current_epoch,
                    dataformats="HWC",
                )
                
                # Генерация и логгирование аудио
                waveform = self.vocoder(output["decoder_outputs"]).cpu().squeeze().numpy()
                trainer.logger.experiment.add_audio(
                    f"test_text_audio/{i}",
                    waveform,
                    trainer.current_epoch,
                    sample_rate=22050,
                ) 