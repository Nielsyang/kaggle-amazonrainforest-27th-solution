import os
import numpy as np
import pandas as pd

PATH_OF_SUBS = './subs'
LABEL_MAP = './label_map.txt'
out = np.zeros((61191,17))
filenames = []
for rt,drs,files in os.walk(PATH_OF_SUBS):
	for file in files:
		if file.startswith('s'):
			filenames.append(os.path.join(rt,file))

def load_label_map(file):
    with open(file, 'r') as f:
        label = f.readlines()
        label = [i[:-2].split(',') for i in label]

    label_map = {int(i[0]):i[1] for i in label}
    inv_label_map = {i[1]:int(i[0]) for i in label}

    return label_map, inv_label_map

inv_label_map, label_map = load_label_map(LABEL_MAP)

submissions = map(lambda x:open(x,'r'), filenames)

for sub in submissions:
	lines = sub.readlines()[1:]
	# print(len(lines))
	cur_out = np.zeros((61191,17))
	for i in range(61191):
		targets = np.zeros(17)

		# if i==0:
		# 	print(lines[i][:-1].split(',')[1].split(' '))
		# 	print('ok')
		for target in lines[i][:-1].split(',')[1].split(' '):
			targets[label_map[target]] = 1

		cur_out[i] = targets
	out += cur_out

out = map(lambda x:x>=len(submissions)/2, out)

output = open('./ensemble.csv', 'w')
output.write('image_name,tags\n')
img_name = pd.read_csv('./submission.csv')['image_name']

for i in range(61191):
	s = [inv_label_map[j] for j in np.where(out[i]==True)[0]]
	output.write(img_name[i]+','+" ".join(s))
	output.write('\n')

for sub in submissions:
	sub.close()
output.close()

