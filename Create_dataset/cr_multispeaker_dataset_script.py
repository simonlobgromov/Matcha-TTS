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

_doc_ = """

!!! This script collects a multispeaker dataset from several recorded by one speaker !!!

"""

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

def get_data_dict(num_spkr:int)->dict:
  dict_ = {}
  # id : dataset_full_name
  for i in range(1, num_spkr+1):
    print(f'--- DATASET {i} / {num_spkr}---')
    dict_[i] = input('Write HF dataset name as <REPO_NAME/DATASET_NAME>: ')
  return dict_


def main()->None:
  print(_doc_, '\n')

  num_spkr = int(input('Write NUM of speakers'))

  if num_spkr > 1:
    data_dict = get_data_dict(num_spkr)
  else:
    print('Use the "create-dataset" script!')
    raise ValueError('NUM of speakers must be more than 1 !!!')


  art = """
            /\_/\ 
           ( o.o ) 
            > ^ <

      V   O   I   C    E
  """
  print(art)
  print()
  print('--- LOADING DATASET ---')

  # LOADING

  for sp_id, dataset_full_name in data_dict.items():
    sub_name_dataset = dataset_full_name.split('/')[1]







  
  os.mkdir(sub_name_dataset)
  os.chdir(sub_name_dataset)
  os.mkdir('wavs')
  os.chdir('wavs')



  your_dataset = load_dataset(name_dataset)
  
  # mk TRAIN
  print()
  print('--- CONVERTIND AND SAVING THE TRAIN DATASET ---')
  num_shards=20
  path = []
  text = []

  with tqdm(total=len(your_dataset['train']), leave=False) as pbar:
    for ind in range(num_shards):
      dataset_shard = your_dataset['train'].shard(num_shards=num_shards, index=ind)
      for row in dataset_shard:
        load_audio(row['audio'])
        path.append(row['audio']['path'])
        text.append(row['raw_transcription'])
        pbar.update(1)

  
  absolute_path = os.path.abspath('../')
  os.chdir(absolute_path)
  
  dir = f'{absolute_path}/wavs/'
  df = pd.DataFrame({'path':path, 'text':text})
  df.text = df.text.map(remove_outer_quotes_regex)
  df.path = dir + df.path
  df.to_csv(f'{sub_name_dataset}_filelist_train.txt', sep='|', header=None, index=False)
  
  # mk TEST
  os.chdir(dir)
  path = []
  text = []
  print()
  print('--- CONVERTIND AND SAVING THE TEST DATASET ---')
  with tqdm(total=len(your_dataset['test']), leave=False) as pbar2:
    for row in tqdm(your_dataset['test']):
      load_audio(row['audio'])
      path.append(row['audio']['path'])
      text.append(row['raw_transcription'])
      pbar2.update(1)
  
  os.chdir(absolute_path)
  df = pd.DataFrame({'path':path, 'text':text})
  df.text = df.text.map(remove_outer_quotes_regex)
  df.path = dir + df.path
  df.to_csv(f'{sub_name_dataset}_filelist_test.txt', sep='|', header=None, index=False)
  print()
  print('--- THE DATASET IS READY ---')
  print(f'Dir of data is "{absolute_path}"')
  
  absolute_path_home = os.path.abspath('../')
  os.chdir(absolute_path_home)


if __name__ == "__main__":
  main()
