import os
import torch
import torch.multiprocessing as mp
import pandas as pd
from transformers import pipeline
import argparse
from tqdm import tqdm

def worker(num_gpus: int, rank: int, model_name: str, file_dir: list, path_list, text_list):
    pipe = pipeline(model=model_name, device=rank)
    len_ = len(file_dir)
    part_list = [i * len_ // num_gpus for i in range(num_gpus)]
    part_list.append(len_)
    file_list = file_dir[part_list[rank]:part_list[rank + 1]]
    local_path_list = []
    local_text_list = []
    for file_ in tqdm(file_list):
        text = pipe(file_)["text"]
        local_path_list.append(file_)
        local_text_list.append(text)
    
    path_list.extend(local_path_list)
    text_list.extend(local_text_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe audio files using multiple GPUs.")
    parser.add_argument('dir_path', type=str, help='Directory path containing audio files')
    args = parser.parse_args()

    os.chdir(args.dir_path)
    file_dir = [f for f in os.listdir() if os.path.isfile(f)]

    manager = mp.Manager()
    path_list = manager.list()
    text_list = manager.list()


    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  

    model_name = "UlutSoftLLC/whisper-small-kyrgyz"
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpus}')
    assert num_gpus >= 2, "Not enough GPUs available"

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(num_gpus, rank, model_name, file_dir, path_list, text_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    absolute_path_home = os.path.abspath('../')
    os.chdir(absolute_path_home)
    pd.DataFrame({'path': list(path_list), 'text': list(text_list)}).to_csv('result.csv', index=False)
