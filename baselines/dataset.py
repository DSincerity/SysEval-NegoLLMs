
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import json
from datasets import load_dataset
from typing import List, Dict
from pathlib import Path
from nego_datasets.negotiation_ji import read_ji_negotiations
from utils import load_jsonl

data_full_nm_mapper= {'cra': 'cra', 'dnd': 'dnd', 'ji': 'job_interview', 'ca': 'casino'}

class RawDatesetLoader(object):

    def __init__(self, dataset_names:List, storage_dir:str):
        """"""
        self.dataset_name = dataset_names
        self.storage_dir = storage_dir

        assert isinstance(self.dataset_name, list), "dataset_names must be a list"
        merged_dataset = dict()

        for dataset in self.dataset_name:
            print('loading ', dataset)
            merged_dataset[data_full_nm_mapper[dataset]] = self.load(dataset)

            if dataset == "ji":
                merged_dataset["job_interview_dialacts"] = self.load_ji_dialacts()
            if dataset == 'dnd':
                merged_dataset['dnd_dialacts'] = self.load_dnd_dialacts()

        self.merged_dataset = merged_dataset

    def load(self, dataset_name:str) -> List[Dict]:
        """"""
        if dataset_name == "ca":
            return self.load_ca()
        elif dataset_name == "cra":
            return self.load_cra()
        elif dataset_name == "dnd":
            return self.load_dnd()
        elif dataset_name == "ji":
            return self.load_ji()
        elif dataset_name == "ji_dialacts":
            return self.load_ji_dialacts()
        elif dataset_name == "dnd_dialacts":
            return self.load_dnd_dialacts()
        else:
            raise ValueError("Invalid dataset name")

    def load_ca(self):
        """"""
        raw_datasets = load_dataset(
                'json',
                data_files={
                    "train": f'{self.storage_dir}/ca.train.jsonl',
                    "test": f'{self.storage_dir}/ca.test.jsonl',
                },
                cache_dir='.cache',
        )
        org_dataset = dict()
        org_dataset['train']= [x for x in raw_datasets['train']]
        org_dataset['test']= [x for x in raw_datasets['test']]
        del raw_datasets
        return org_dataset

    def load_ji(self):
        org_dataset = dict()
        org_dataset['train'] = read_ji_negotiations(os.path.join(self.storage_dir, "ji.train.json"))
        org_dataset['test'] = read_ji_negotiations(os.path.join(self.storage_dir, "ji.test.json"))
        return org_dataset

    def load_ji_dialacts(self):
        return self.load_jsonl("ji", "dialacts.")

    def load_dnd_dialacts(self):
        return self.load_json("dnd.sample", "dialacts.")

    def load_dnd(self):
        return self.load_json("dnd.sample")

    def load_cra(self):
        return self.load_json("cra")

    def load_json(self, datset_nm, postfix=""):
        org_dataset = dict()
        org_dataset['train'] = json.loads(Path(os.path.join(self.storage_dir, f"{datset_nm}.train.{postfix}json")).read_text())
        org_dataset['test'] = json.loads(Path(os.path.join(self.storage_dir, f"{datset_nm}.test.{postfix}json")).read_text())
        return org_dataset

    def load_jsonl(self, datset_nm, postfix=""):
        org_dataset = dict()
        org_dataset['train'] = load_jsonl(os.path.join(self.storage_dir, f"{datset_nm}.train.{postfix}jsonl"))
        org_dataset['test'] = load_jsonl(os.path.join(self.storage_dir, f"{datset_nm}.test.{postfix}jsonl"))
        return org_dataset

if __name__=="__main__":

    #############################################################
    ## split train into train and test with transformers dataset
    ############################################################

    dataset_id = "casino"
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(dataset)

    # split train into train and test
    dataset = dataset['train'].train_test_split(test_size=0.1)

    # save
    # save_dir='./datasets'
    # for split, dataset in dataset.items():
    #     dataset.to_json(os.path.join(save_dir, f"casino.{split}.jsonl"))

    ################################################################
    # raw dataset loader
    ################################################################

    raw_dataset = RawDatesetLoader(['ca', 'dnd', 'cra', 'ji'], storage_dir='baselines/datasets').merged_dataset
    print(raw_dataset.keys())
    print("="*10, 'ca', "="*10)
    print(raw_dataset['casino']['train'][0])
    print(raw_dataset['casino']['test'][0])
    print(len(raw_dataset['casino']['train']), len(raw_dataset['casino']['test']))
    print()

    print("="*10, 'dnd', "="*10)
    print(raw_dataset['dnd']['train'][0])
    print(raw_dataset['dnd']['test'][0])
    print(len(raw_dataset['dnd']['train']), len(raw_dataset['dnd']['test']))
    print()

    print("="*10, 'dnd dialacts', "="*10)
    print(raw_dataset['dnd_dialacts']['train'][0])
    print(raw_dataset['dnd_dialacts']['test'][0])
    print(len(raw_dataset['dnd_dialacts']['train']), len(raw_dataset['dnd_dialacts']['test']))

    print("="*10, 'cra', "="*10)
    print(raw_dataset['cra']['train'][0])
    print(raw_dataset['cra']['test'][0])
    print(len(raw_dataset['cra']['train']), len(raw_dataset['cra']['test']))
    print()

    print("="*10, 'job_interview', "="*10)
    print(raw_dataset['job_interview']['train'][0])
    print(raw_dataset['job_interview']['test'][0])
    print(len(raw_dataset['job_interview']['train']), len(raw_dataset['job_interview']['test']))
    print()

    print("="*10, 'job_interview dialacts', "="*10)
    print(raw_dataset['job_interview_dialacts']['train'][0])
    print(raw_dataset['job_interview_dialacts']['test'][0])
    print(len(raw_dataset['job_interview_dialacts']['train']), len(raw_dataset['job_interview_dialacts']['test']))
