import pandas as pd
import numpy as np
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from IPython.display import Audio
import scipy
import librosa
from tqdm import tqdm
import re
import os
import argparse


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

def main(data_rep)->None:
  os.mkdir('kany_dataset')
  os.chdir('kany_dataset')
  os.mkdir('wavs')
  os.chdir('wavs')


  art = """
            /\_/\ 
           ( o.o ) 
            > ^ <

      V   O   I   C    E
  """
  print(art)
  
  print('--- LOADING DATASET ---')
  dataset_kany = load_dataset(data_rep)
  
  # mk TRAIN
  print()
  print('--- CONVERTIND AND SAVING THE TRAIN DATASET ---')
  num_shards=20
  path = []
  text = []
  for ind in tqdm(range(num_shards)):
    dataset_shard = dataset_kany['train'].shard(num_shards=num_shards, index=ind)
    for row in tqdm(dataset_shard):
      load_audio(row['audio'])
      path.append(row['audio']['path'])
      text.append(row['raw_transcription'])
  
  absolute_path = os.path.abspath('../Matcha-TTS/kany_dataset')
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
  print()
  print('--- CONVERTIND AND SAVING THE TEST DATASET ---')
  for row in tqdm(dataset_kany['test']):
    load_audio(row['audio'])
    path.append(row['audio']['path'])
    text.append(row['raw_transcription'])
  
  os.chdir(absolute_path)
  df = pd.DataFrame({'path':path, 'text':text})
  df.text = df.text.map(remove_outer_quotes_regex)
  df.path = dir + df.path
  df.to_csv('kany_filelist_test.txt', sep='|', header=None, index=False)
  print()
  print('--- THE DATASET IS READY ---')
  print(f'Dir of data is "{absolute_path}"')
  
  absolute_path_home = os.path.abspath('../Matcha-TTS')
  os.chdir(absolute_path_home)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_rep", type=str, help="HF-Dataset representation in format 'author/dataset_name'")
    args = parser.parse_args()
    main(args.data_rep)
