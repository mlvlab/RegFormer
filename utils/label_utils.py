"""
Utility functions for loading text labels from different datasets
"""
import sys
import os

# Add utils_tip to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_tip_path = os.path.join(parent_dir, 'utils_tip')
if utils_tip_path not in sys.path:
    sys.path.append(utils_tip_path)

try:
    from utils.hico_text_label import hico_text_label, hico_obj_text_label, hico_hum_obj_text_label, hico_hum_obj_text_label_2
    from hico_list import hico_verbs_sentence
    from vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
    from vcoco_list import vcoco_verbs_sentence
except ImportError as e:
    print(f"Warning: Could not import label files: {e}")
    print("Please ensure utils_tip directory is accessible")
    # Fallback empty data
    hico_text_label = {}
    hico_verbs_sentence = []
    vcoco_hoi_text_label = {}
    vcoco_verbs_sentence = []


def get_verb_object_indices(dataset_name):
    if dataset_name == 'hico':
        return ([k[0] for k,v in hico_text_label.items()], [k[1] for k,v in hico_text_label.items()])
    elif dataset_name == 'vcoco':
        return ([k[0] for k,v in vcoco_hoi_text_label.items()], [k[1]-1 for k,v in vcoco_hoi_text_label.items()]) # vcoco object index starts from 1
    elif dataset_name == 'swig':
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swig_hoi.swig_v1_categories import SWIG_INTERACTIONS
        sorted_interactions = sorted(SWIG_INTERACTIONS, key=lambda x: x['id'])
        return ([interaction['action_id'] for interaction in sorted_interactions], [interaction['object_id'] for interaction in sorted_interactions])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_class_labels(dataset_name, target_type):
    """
    Get class text labels based on dataset and target type
    
    Args:
        dataset_name: 'hico' or 'vcoco'
        target_type: 'hoi' or 'verb'
    
    Returns:
        labels: List of text labels
        num_classes: Number of classes
    """
    dataset_name = dataset_name.lower()
    target_type = target_type.lower()
    
    if dataset_name == 'hico':
        if target_type == 'hoi':
            # Convert hico_text_label dict to list
            # Keys are (verb_id, object_id), values are text descriptions
            labels = []
            max_key = max(hico_text_label.keys()) if hico_text_label else (0, 0)
            # Create mapping from (verb_id, object_id) to index
            for i, (key, text) in enumerate(hico_text_label.items()):
                labels.append(text)
            num_classes = len(labels)
            
        elif target_type == 'verb':
            labels = hico_verbs_sentence
            num_classes = len(labels)
        elif target_type == 'object':
            labels = []
            for (_, text) in hico_obj_text_label:
                labels.append(text)
            num_classes = len(labels)
        
        elif target_type == 'hum_interact_obj':
            labels = []
            for (_, text) in hico_hum_obj_text_label:
                labels.append(text)
            num_classes = len(labels)
        elif target_type == 'hum_and_obj':
            labels = []
            for (_, text) in hico_hum_obj_text_label_2:
                labels.append(text)
            num_classes = len(labels)
        else:
            raise ValueError(f"Unsupported target_type for HICO: {target_type}")
            
    elif dataset_name == 'vcoco':
        if target_type == 'hoi':
            # Convert vcoco_hoi_text_label dict to list
            labels = []
            for i, (key, text) in enumerate(vcoco_hoi_text_label.items()):
                labels.append(text)
            num_classes = len(labels)
            
        elif target_type == 'verb':
            labels = vcoco_verbs_sentence
            num_classes = len(labels)
        elif target_type == 'object':
            labels = []
            for (_, text) in vcoco_obj_text_label:
                labels.append(text)
            num_classes = len(labels)
            
        else:
            raise ValueError(f"Unsupported target_type for V-COCO: {target_type}")
            
    elif dataset_name == 'swig':
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swig_hoi.swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES
        template = "a photo of a"
        formatting = lambda x: f"{template} {x}"
        if target_type == 'hoi':
            labels = []
            sorted_interactions = sorted(SWIG_INTERACTIONS, key=lambda x: x['id'])
            for interaction in sorted_interactions:
                labels.append(formatting(f"person {interaction['name']}"))
            num_classes = len(labels)
        elif target_type == 'verb':
            labels = []
            sorted_actions = sorted(SWIG_ACTIONS, key=lambda x: x['id'])
            for action in sorted_actions:
                labels.append(formatting(f"person {action['name']} the object"))
            num_classes = len(labels)
        elif target_type == 'object':
            labels = []
            sorted_categories = sorted(SWIG_CATEGORIES, key=lambda x: x['id'])
            for category in sorted_categories:
                labels.append(formatting(f"{category['name']}"))
            num_classes = len(labels)
        else:
            raise ValueError(f"Unsupported target_type for SWIG: {target_type}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return labels, num_classes

def verb_to_hoi_dict(dataset_name):
    if dataset_name == 'hico':
        verb_to_hoi_dict = {}
        for i,k in enumerate(hico_text_label.keys()):
            if k[0] not in verb_to_hoi_dict:
                verb_to_hoi_dict[k[0]] = []
            verb_to_hoi_dict[k[0]].append(i)
        return verb_to_hoi_dict
    elif dataset_name == 'vcoco':
        verb_to_hoi_dict = {}
        for i,k in enumerate(vcoco_hoi_text_label.keys()):
            if k[0] not in verb_to_hoi_dict:
                verb_to_hoi_dict[k[0]] = []
            verb_to_hoi_dict[k[0]].append(i)
        return verb_to_hoi_dict
    elif dataset_name == 'swig':
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swig_hoi.swig_v1_categories import SWIG_INTERACTIONS
        verb_to_hoi_dict = {}
        for interaction in SWIG_INTERACTIONS:
            if interaction['action_id'] not in verb_to_hoi_dict:
                verb_to_hoi_dict[interaction['action_id']] = []
            verb_to_hoi_dict[interaction['action_id']].append(interaction['id'])
        return verb_to_hoi_dict
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def hoi_to_verb_list(dataset_name):
    if dataset_name == 'hico':
        hoi_to_verb_list = []
        for i,k in enumerate(hico_text_label.keys()):
            hoi_to_verb_list.append(k[0])
        return hoi_to_verb_list
    elif dataset_name == 'vcoco':
        hoi_to_verb_list = []
        for i,k in enumerate(vcoco_hoi_text_label.keys()):
            hoi_to_verb_list.append(k[0])
        return hoi_to_verb_list
    elif dataset_name == 'swig':
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swig_hoi.swig_v1_categories import SWIG_INTERACTIONS
        sorted_interactions = sorted(SWIG_INTERACTIONS, key=lambda x: x['id'])
        return [interaction['action_id'] for interaction in sorted_interactions]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_label_mapping(dataset_name, target_type):
    """
    Create mapping from original IDs to our class indices
    
    Args:
        dataset_name: 'hico' or 'vcoco'
        target_type: 'hoi' or 'verb'
    
    Returns:
        mapping: Dict mapping original IDs to class indices
    """
    dataset_name = dataset_name.lower()
    target_type = target_type.lower()
    
    mapping = {}
    
    if dataset_name == 'hico':
        if target_type == 'hoi':
            # For HOI, mapping from (verb_id, object_id) to class index
            for i, key in enumerate(hico_text_label.keys()):
                mapping[key] = i
                
        elif target_type == 'verb':
            # For verb, direct mapping from verb_id to class index
            for i in range(len(hico_verbs_sentence)):
                mapping[i] = i
                
    elif dataset_name == 'vcoco':
        if target_type == 'hoi':
            # For HOI, mapping from (verb_id, object_id) to class index
            for i, key in enumerate(vcoco_hoi_text_label.keys()):
                mapping[key] = i
                
        elif target_type == 'verb':
            # For verb, direct mapping from verb_id to class index
            for i in range(len(vcoco_verbs_sentence)):
                mapping[i] = i
    
    return mapping


def print_label_info(dataset_name, target_type):
    """
    Print information about the loaded labels
    """
    labels, num_classes = get_class_labels(dataset_name, target_type)
    mapping = create_label_mapping(dataset_name, target_type)
    
    print(f"\n=== Label Info for {dataset_name.upper()} {target_type.upper()} ===")
    print(f"Number of classes: {num_classes}")
    print(f"First 5 labels:")
    for i, label in enumerate(labels[:5]):
        print(f"  {i}: {label}")
    print(f"Last 3 labels:")
    for i, label in enumerate(labels[-3:], start=len(labels)-3):
        print(f"  {i}: {label}")
    print(f"Mapping sample: {dict(list(mapping.items())[:3])}")
    print("=" * 50)


# Test the functions if run directly
if __name__ == "__main__":
    # Test all combinations
    for dataset in ['hico', 'vcoco']:
        for target in ['verb', 'hoi']:
            try:
                print_label_info(dataset, target)
            except Exception as e:
                print(f"Error with {dataset} {target}: {e}")
