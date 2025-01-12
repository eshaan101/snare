import os
import json
import torch
import torch.utils.data
import numpy as np
import gzip

class CLIPGraspingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode='train'):
        self.total_views = 14
        self.cfg = cfg
        self.mode = mode
        self.folds = os.path.join(self.cfg['data']['amt_data'], self.cfg['data']['folds'])
        self.feats_backbone = self.cfg['train']['feats_backbone']

        print(f"Initializing CLIPGraspingDataset in {self.mode} mode.")
        self.load_entries()
        self.load_extracted_features()
        print(f"Dataset initialization complete for {self.mode} mode.")

    def load_entries(self):
        print("Loading entries in CLIPGraspingDataset...")
        train_train_files = ["train.json"]
        train_val_files = ["val.json"]
        test_test_files = ["test.json"]

        # Modes
        if self.mode == "train":
            self.files = train_train_files
        elif self.mode == "valid":
            self.files = train_val_files
        elif self.mode == "test":
            self.files = test_test_files
        else:
            raise RuntimeError('Mode not recognized, should be train, valid or test: ' + str(self.mode))
        
        # Load AMT data
        self.data = []
        for file in self.files:
            fname_rel = os.path.join(self.folds, file)
            print(f"Loading file: {fname_rel}")
            with open(fname_rel, 'r') as f:
                entries = json.load(f)
                print(f"Loaded {len(entries)} entries from {file}")
                self.data += entries

        print(f"Loaded total entries. {self.mode}: {len(self.data)} entries")

    def load_extracted_features(self):
        print("Loading extracted features...")
        if self.feats_backbone == "clip":
            # Language features
            lang_feats_path = self.cfg['data']['clip_lang_feats']
            print(f"Loading language features from: {lang_feats_path}")
            with gzip.open(lang_feats_path, 'rt', encoding='utf-8') as f:
                self.lang_feats = json.load(f)
            print(f"Loaded language features. Total entries: {len(self.lang_feats)}")
    
            # Image features
            img_feats_path = self.cfg['data']['clip_img_feats']
            print(f"Loading image features from: {img_feats_path}")
            with gzip.open(img_feats_path, 'rt', encoding='utf-8') as f:
                self.img_feats = json.load(f)
            print(f"Loaded image features. Total keys: {len(self.img_feats)}")
    
            # Apply subsetting for debugging
            subset_size = 100
            self.img_feats = dict(list(self.img_feats.items())[:subset_size])
            print(f"Using subset of image features. Total keys: {len(self.img_feats)}")
        else:
            raise NotImplementedError("Unsupported feats_backbone. Expected 'clip'.")



    def __len__(self):
        return len(self.data)

    def get_img_feats(self, key):
        feats = []
        for i in range(self.total_views):
            feat_key = f'{key}-{i}'
            feat = np.array(self.img_feats[feat_key])
            feats.append(feat)
        return np.array(feats)

    def __getitem__(self, idx):
        print(f"Fetching item at index {idx}...")
        entry = self.data[idx]

        # Get keys
        entry_idx = entry['ans'] if 'ans' in entry else -1  # Test set does not contain answers
        if len(entry['objects']) == 2:
            key1, key2 = entry['objects']
        else:
            key1 = entry['objects'][entry_idx]
            while True:
                key2 = np.random.choice(list(self.img_feats.keys())).split("-")[0]
                if key2 != key1:
                    break

        # Annotation and features
        annotation = entry['annotation']
        is_visual = entry['visual'] if 'ans' in entry else -1  # Test set does not have labels

        # Feats
        start_idx = 6  # Discard first 6 views
        img1_n_feats = torch.from_numpy(self.get_img_feats(key1))[start_idx:]
        img2_n_feats = torch.from_numpy(self.get_img_feats(key2))[start_idx:]
        lang_feats = torch.from_numpy(np.array(self.lang_feats[annotation]))

        # Label
        ans = entry_idx

        print(f"Successfully fetched item at index {idx}.")
        return (
            (img1_n_feats, img2_n_feats),
            lang_feats,
            ans,
            (key1, key2),
            annotation,
            is_visual,
        )
