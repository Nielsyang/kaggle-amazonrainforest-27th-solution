def load_label_map(file):
    with open(file, 'r') as f:
        label = f.readlines()
        label = [i[:-2].split(',') for i in label]

    label_map = {int(i[0]):i[1] for i in label}
    inv_label_map = {i[1]:int(i[0]) for i in label}

    return label_map, inv_label_map