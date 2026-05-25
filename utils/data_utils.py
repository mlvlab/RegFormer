import torch
import json
import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.hico_text_label import hico_unseen_index

class WeakDATA(Dataset):
    """
    Weak supervision dataset for HOI detection
    """
    
    def __init__(self, args, transforms=None, is_train=True):
        """
        Initialize WeakDATA dataset
        
        Args:
            args: Arguments containing data_path, image_root, target_type, etc.
            transforms: Optional image transforms
        """
        self.args = args
        self.dataset_name = args.dataset_name
        self.transforms = transforms
        self.image_root = args.image_root
        self.target_type = args.target_type
        self.verb_num_classes = args.verb_num_classes
        self.hoi_num_classes = args.hoi_num_classes
        if self.args.dataset_name=='hico':
            self.num_object_classes = 81
        elif self.args.dataset_name=='vcoco':
            self.num_object_classes = 80
        elif self.args.dataset_name=='swig':
            self.num_object_classes = 1000
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset_name}")
        # Load data from JSON file
        print(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r') as f:
            self.data = json.load(f)
        
        if self.args.zs_type is not None and args.target_type=='hoi' and 'hico' in args.dataset_name:
            self.unseen_index = hico_unseen_index[self.args.zs_type]
            # self.remain_index = [i for i in range(self.num_classes) if i not in self.unseen_index]
        else:
            self.unseen_index = None
            # self.remain_index = range(self.num_classes)
        
        # Extract filenames and annotations
        self.filenames = self.data['filenames'] if 'filenames' in self.data else self.data['images']
        self.annotations = self.data['annotation'] if 'annotation' in self.data else self.data['annotations']
        
        if args.target_type == 'verb':
            self.num_classes = self.verb_num_classes
        elif args.target_type == 'hoi':
            self.num_classes = self.hoi_num_classes
        else:
            raise ValueError(f"Unsupported target type: {args.target_type}")
        
        self.label_mask = torch.ones(self.num_classes, dtype=torch.bool)
        if is_train:
            if self.unseen_index is not None:
                self.idx = self.filter_data(self.unseen_index)
                self.label_mask[self.unseen_index] = False
            else:
                self.idx = range(len(self.filenames))
        else:
            self.idx = range(len(self.filenames))
    
        self.is_train = is_train
        # Verify data consistency
        assert len(self.filenames) == len(self.annotations), \
            f"Mismatch between filenames ({len(self.filenames)}) and annotations ({len(self.annotations)})"
        
        print(f"Loaded {len(self.filenames)} samples")
        print(f"Target type: {self.target_type}")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.idx)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: PIL Image or transformed image
            target: Multi-label target vector
        """
        # Get filename and load image
        if self.dataset_name == 'hico':
            filename = self.filenames[self.idx[idx]]
        elif self.dataset_name in ['vcoco', 'swig']:
            filename = self.annotations[self.idx[idx]]['file_name']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        image_path = os.path.join(self.image_root, filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        targets = {}
        # Get annotation for this sample
        annotation = self.annotations[self.idx[idx]]
        
        # # Extract target labels based on target_type
        # if self.target_type == 'verb':
        #     labels = annotation['verb'] if 'verb' in annotation else annotation['actions']
            
        #     # Create multi-label target matcies shape with [num_objects, num_verbs]
        #     target = torch.zeros(self.num_object_classes, self.verb_num_classes, dtype=torch.float32)
        #     for object_label in annotation['object']:
        #         for verb_label in labels:
        #             ol = object_label - 1 if self.dataset_name == 'vcoco' else object_label
        #             vl = verb_label
        #             if 0 <= ol < self.num_object_classes and 0 <= vl < self.verb_num_classes:
        #                 target[ol, vl] = 1.0
        #             else:
        #                 print(f"Warning: Label {ol} or {vl} is out of range [0, {self.num_object_classes-1}] or [0, {self.verb_num_classes-1}]")
                        
        # elif self.target_type == 'hoi':
        #     labels = annotation['hoi']
        #     # Remove duplicates by converting to set
        #     unique_labels = list(set(labels))
            
        #     # Create multi-label target vector
        #     target = torch.zeros(self.hoi_num_classes, dtype=torch.float32)
            
        #     # Set 1 for present classes
        #     for label in unique_labels:
        #         if self.is_train and self.args.zs_type is not None and label in self.unseen_index:
        #             continue
        #         if 0 <= label < self.hoi_num_classes:
        #             target[label] = 1.0
        #         else:
        #             print(f"Warning: Label {label} is out of range [0, {self.hoi_num_classes-1}]")
        # else:
        #     raise ValueError(f"Unsupported target type: {self.target_type}")
        
        # verb targets
        labels = annotation['verb'] if 'verb' in annotation else annotation['actions']
        
        # Create multi-label target matcies shape with [num_objects, num_verbs]
        target = torch.zeros(self.num_object_classes, self.verb_num_classes, dtype=torch.float32)
        for object_label in annotation['object']:
            for verb_label in labels:
                ol = object_label - 1 if self.dataset_name == 'vcoco' else object_label
                vl = verb_label
                if 0 <= ol < self.num_object_classes and 0 <= vl < self.verb_num_classes:
                    target[ol, vl] = 1.0
                else:
                    print(f"Warning: Label {ol} or {vl} is out of range [0, {self.num_object_classes-1}] or [0, {self.verb_num_classes-1}]")
        
        targets['verb'] = target
        
        # hoi targets
        labels = annotation['hoi']
        # Remove duplicates by converting to set
        unique_labels = list(set(labels))
        
        # Create multi-label target vector
        target = torch.zeros(self.hoi_num_classes, dtype=torch.float32)
        
        # Set 1 for present classes
        for label in unique_labels:
            if self.is_train and self.args.zs_type is not None and label in self.unseen_index:
                continue
            if 0 <= label < self.hoi_num_classes:
                target[label] = 1.0
            else:
                print(f"Warning: Label {label} is out of range [0, {self.hoi_num_classes-1}]")
        
        targets['hoi'] = target
        
        # add interactiveness target
        if self.args.use_seperate_interactiveness_loss:
            interactiveness_target = torch.zeros(self.num_object_classes, dtype=torch.float32)
            for object_label in annotation['object']:
                # if self.is_train and self.args.zs_type is not None and label in self.unseen_index:
                #     continue
                if 0 <= object_label < self.num_object_classes:
                    interactiveness_target[object_label] = 1.0
            targets['interactiveness'] = interactiveness_target
        
        # Apply transforms if provided
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, targets
    
    def filter_data(self, unseen_index):
        """
        Filter data based on unseen index
        """
        remain_index = [i for i in range(self.num_classes) if i not in unseen_index]
        idx = []
        for i in range(len(self.filenames)):
            mutual_hoi = set(remain_index) & set(self.annotations[i]['hoi'])
            if len(mutual_hoi) != 0:
                idx.append(i)
        return idx
    
    def get_annotation_info(self, idx):
        """
        Get full annotation information for debugging
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict: Full annotation information
        """
        annotation = self.annotations[self.idx[idx]].copy()
        annotation['filename'] = self.filenames[self.idx[idx]]
        return annotation
    
    def get_class_statistics(self):
        """
        Get statistics about class distribution
        
        Returns:
            dict: Class statistics
        """
        class_counts = torch.zeros(self.num_classes)
        total_unique_labels = 0
        
        for i in self.idx:
            annotation = self.annotations[i]
            if self.target_type == 'verb':
                labels = annotation['verb'] if 'verb' in annotation else annotation['actions']
            else:
                labels = annotation['hoi']
            
            # Remove duplicates within each sample
            unique_labels = list(set(labels))
            total_unique_labels += len(unique_labels)
            
            for label in unique_labels:
                if self.args.zs_type is not None and label in self.unseen_index:
                    continue
                if 0 <= label < self.num_classes:
                    class_counts[label] += 1
        
        return {
            'class_counts': class_counts,
            'total_samples': len(self.idx),
            'classes_with_samples': (class_counts > 0).sum().item(),
            'avg_unique_labels_per_sample': total_unique_labels / len(self.idx)
        }
