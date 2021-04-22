import torch
import torch.utils.data
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from data import Fragment

# This file is for spectrogram visualization by Zhang
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

Files=load_variable("/home/zzhang/test/file.data")
for file in Files:
	if file.label[0]==1:
		librosa.display.specshow(librosa.amplitude_to_db(file.spec,ref=np.max),y_axis='log', x_axis='time')
		plt.title('happy')
		plt.tight_layout()
		plt.savefig('hap.png')
		break

for file in Files:
	if file.label[1]==1:
		librosa.display.specshow(librosa.amplitude_to_db(file.spec,ref=np.max),y_axis='log', x_axis='time')
		plt.title('neutral')
		plt.tight_layout()
		plt.savefig('neu.png')
		break

for file in Files:
	if file.label[2]==1:
		librosa.display.specshow(librosa.amplitude_to_db(file.spec,ref=np.max),y_axis='log', x_axis='time')
		plt.title('sad')
		plt.tight_layout()
		plt.savefig('sad.png')
		break

for file in Files:
	if file.label[3]==1:
		librosa.display.specshow(librosa.amplitude_to_db(file.spec,ref=np.max),y_axis='log', x_axis='time')
		plt.title('angry')
		plt.tight_layout()
		plt.savefig('ang.png')
		break

for file in Files:
	if file.label[4]==1:
		librosa.display.specshow(librosa.amplitude_to_db(file.spec,ref=np.max),y_axis='log', x_axis='time')
		plt.title('frustrated')
		plt.tight_layout()
		plt.savefig('fru.png')
		break
