import pandas as pd
import numpy as np
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from IPython.display import Audio
import scipy
import librosa
from tqdm import tqdm_notebook
import re
import os

def load_audio(audio_dict:dict)->None:
  target_sr = 22050
  audio_resampled = librosa.resample(np.array(audio_dict['array']),
                                     orig_sr=audio_dict['sampling_rate'],
                                     target_sr=target_sr)
  scipy.io.wavfile.write(audio_dict['path'],
                         rate=target_sr,
                         data=(audio_resampled* 32767).astype(np.int16))

def remove_outer_quotes_regex(sen:str)->str:
  return re.sub(r'^["\'](.*)["\']$', r'\1', sen)

def main()->None:
  os.mkdir('kany_dataset')
  os.chdir('kany_dataset')
  os.mkdir('wavs')
  os.chdir('wavs')
  
  print('--- LOADING DATASET ---')
  dataset_kany = load_dataset("Simonlob/Kany_dataset_mk4")
  
  # mk TRAIN
  print('--- CONVERTIND AND SAVING THE TRAIN DATASET ---')
  num_shards=20
  path = []
  text = []
  for ind in tqdm_notebook(range(num_shards)):
    dataset_shard = dataset_kany['train'].shard(num_shards=num_shards, index=ind)
    for row in tqdm_notebook(dataset_shard):
      load_audio(row['audio'])
      path.append(row['audio']['path'])
      text.append(row['raw_transcription'])
  
  absolute_path = os.path.abspath('../Matcha/kany_dataset')
  os.chdir(absolute_path)
  
  dir = f'{absolute_path}/wavs/'
  df = pd.DataFrame({'path':path, 'text':text})
  df.text = df.text.map(remove_outer_quotes_regex)
  df.path = dir + df.path
  df.to_csv('kany_filelist_train.txt', sep='|', header=None, index=False)
  
  # mk TEST
  os.chdir(dir)
  path = []
  text = []
  print('--- CONVERTIND AND SAVING THE TEST DATASET ---')
  for row in tqdm_notebook(dataset_kany['test']):
    load_audio(row['audio'])
    path.append(row['audio']['path'])
    text.append(row['raw_transcription'])
  
  os.chdir(absolute_path)
  df = pd.DataFrame({'path':path, 'text':text})
  df.text = df.text.map(remove_outer_quotes_regex)
  df.path = dir + df.path
  df.to_csv('kany_filelist_test.txt', sep='|', header=None, index=False)
  print('--- THE DATASET IS READY ---')
  print(f'Dir of data is "{absolute_path}"')
  
  absolute_path = os.path.abspath('../Matcha')
  os.chdir(absolute_path)

if __name__ == "__main__":
    main()
