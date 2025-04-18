from config import mapping_dir
import os, csv
import math

train_map = {}
test_map = {}
val_map = {}

def initialize_map(t_type):
    global train_map, test_map, val_map
    
    if t_type == 'train':
        train_map = read_file('train', mapping_dir)
    elif t_type == 'test':
        test_map  = read_file('test',  mapping_dir)
    else:
        val_map   = read_file('val',   mapping_dir)

def get_mapping(patient_name, t_type):
    if t_type == 'train':
        if not train_map:
            initialize_map('train')
        return train_map.get(patient_name)
    elif t_type == 'test':
        if not test_map:
            initialize_map('test')
        return test_map.get(patient_name)
    else:
        if not val_map:
            initialize_map('val')
        return val_map.get(patient_name)

def get_theta(input_fov, target_fov, eye):
    fov_tuple = (input_fov, target_fov)
    se2_lookup = {
        ("OD", "Nasal"): (math.pi/6),
        ("OD", "Temporal"): (-math.pi/6)
    }
    theta = se2_lookup[fov_tuple]
    if eye == "right":
        theta *= -1
    return theta

def read_file(t_type: str, mapping_dir: str) -> dict[str, tuple[str, str]]:
    file_map = {
        'train': '/patient_id_mapping_train.csv',
        'test' : '/patient_id_mapping_test.csv',
        'val'  : '/patient_id_mapping_val.csv'
    }
    
    try:
        fname = file_map[t_type]
    except KeyError:
        raise ValueError(f"Unknown type '{t_type}'; expected one of {list(file_map)}")

    path = mapping_dir + fname
    mapping: dict[str, tuple[str, str]] = {}
    
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['patient_id']
            mapping[pid] = (row['side_eye'], row['eye_text'])
            
    return mapping