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

class Data:

  def __init__(self, repo_name:str):
    self.repo_name = repo_name

  @staticmethod
  def load_audio(audio_dict:dict)->None:
    target_sr = 22050
    audio_resampled = librosa.resample(np.array(audio_dict['array']),
                                      orig_sr=audio_dict['sampling_rate'],
                                      target_sr=target_sr)
    scipy.io.wavfile.write(audio_dict['path'],
                          rate=target_sr,
                          data=(audio_resampled* 32767).astype(np.int16))

  @staticmethod
  def remove_outer_quotes_regex(sen:str)->str:
    return re.sub(r'^["\'](.*)["\']$', r'\1', sen)

  def load_data(self)->None:
    self.dataset = load_dataset(self.repo_name)

    print('--- CONVERTIND AND SAVING THE TRAIN DATASET ---')
    num_shards=20
    path = []
    text = []

    os.mkdir(self.repo_name.split('/')[-1])
    os.chdir(self.repo_name.split('/')[-1])
    
    with tqdm(total=len(self.dataset['train']), leave=False) as pbar:
      for ind in range(num_shards):
        dataset_shard = self.dataset['train'].shard(num_shards=num_shards, index=ind)
        for row in dataset_shard:
          self.load_audio(row['audio'])
          path.append(row['audio']['path'])
          text.append(row['raw_transcription'])
          pbar.update(1)


    df_train = pd.DataFrame({'path':path, 'text':text})
    df_train.text = df_train.text.map(self.remove_outer_quotes_regex)
    df_train.path = os.path.abspath('./') + '/' + df_train.path
    
    # mk TEST
    path = []
    text = []
    print()
    print('--- CONVERTIND AND SAVING THE TEST DATASET ---')
    with tqdm(total=len(self.dataset['test']), leave=False) as pbar2:
      for row in tqdm(self.dataset['test']):
        self.load_audio(row['audio'])
        path.append(row['audio']['path'])
        text.append(row['raw_transcription'])
        pbar2.update(1)
    
    df_test = pd.DataFrame({'path':path, 'text':text})
    df_test.text = df_test.text.map(self.remove_outer_quotes_regex)
    df_test.path = os.path.abspath('./') + '/' + df_test.path
    print(f'--- THE DATASET IS READY ---')
    
    absolute_path_home = os.path.abspath('../')
    os.chdir(absolute_path_home)

    self.file_lists = df_train, df_test




def get_data_dict(num_spkr:int)->dict:
  dict_ = {}
  # id : dataset_full_name
  for i in range(1, num_spkr+1):
    print(f'--- DATASET {i} / {num_spkr} ---')
    dict_[i] = input('Write HF dataset name as <REPO_NAME/DATASET_NAME>: ')
  return dict_


def main()->None:
  os.mkdir('akylai_multi_dataset')
  os.chdir('akylai_multi_dataset')
  print(_doc_, '\n')

  num_spkr = int(input('Write NUM of speakers: '))

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
  print('--- LOADING DATASETS ---', '\n')

  # LOADING
  filelist_dict = {}
  for dataset_full_name in set(data_dict.values()):
    print(f'-- loading {dataset_full_name} ---')
    sub_name_dataset = dataset_full_name.split('/')[1]
    filelist_dict[sub_name_dataset] = Data(dataset_full_name)
    filelist_dict[sub_name_dataset].load_data()


  df_train = pd.DataFrame(columns=['path', 'sp_id', 'text'])
  df_test = pd.DataFrame(columns=['path', 'sp_id', 'text'])
  for sp_id, repo_name in data_dict.items():
    train = filelist_dict[repo_name.split('/')[-1]].file_lists[0]
    test = filelist_dict[repo_name.split('/')[-1]].file_lists[1]
    train['sp_id'] = sp_id
    test['sp_id'] = sp_id
    train = train[['path', 'sp_id', 'text']]
    test = test[['path', 'sp_id', 'text']]
    df_train = pd.concat([df_train, train], axis=0)
    df_test = pd.concat([df_test, test], axis=0)

  df_train.to_csv('akylai_mlspk_filelist_train.txt', sep='|', header=None, index=False)
  df_test.to_csv('akylai_mlspk_filelist_test.txt', sep='|', header=None, index=False)

  absolute_path_home = os.path.abspath('../')
  os.chdir(absolute_path_home)

  print(art)
    



if __name__ == "__main__":
  main()
