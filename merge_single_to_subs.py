import pandas as pd
import os
import pdb

sub = pd.DataFrame()

subs = []
for rt,dirs,files in os.walk('/home/yy/kaggle_pytorch/subs_single'):
	for file in files:
		subs.append(pd.read_csv(os.path.join(rt,file)))
merge_num = len(subs)

sub['image_name'] = subs[0]['image_name']
tags = ['']*len(sub)
for i in range(len(tags)):
	times_pos = 0
	for s in subs:
		if s['tags'][i] == 'water':
			times_pos += 1
	tags[i] = 'water' if times_pos>merge_num/2 else 'n'
sub['tags'] = tags
sub.to_csv('./ensemble_6.csv', index=False)
