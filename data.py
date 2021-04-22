import torch
import torch.utils.data
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

# Read all files, size of dataset:3251
# generate test train anad validate dataset
#Execute this function to generate .data files
def readFiles():
    Files=[]
    test=[]
    valid=[]
    train=[]
    Sex='F'
    print('reading session 1')
    for Index in [1,2,3,4,5,6,7]:
        Files.extend(readOneFiles(1, Index, 'F', ''))
        Files.extend(readOneFiles(1, Index, 'M', ''))
    for Session in [2,4,5]:
        print('reading session '+str(Session))
        Sex='F'
        for Index in [1,2,3,4,5,6,7,8]:
            Files.extend(readOneFiles(Session, Index, 'F', ''))
            Files.extend(readOneFiles(Session, Index, 'M', ''))
    Session=3
    print('reading session '+str(Session))
    for Index in [1,2,3,4,5,6,7,8]:
        Files.extend(readOneFiles(Session, Index, 'F', ''))
    for Index in [1,2,3,4,6,7]:
        Files.extend(readOneFiles(Session, Index, 'M', ''))
    for Index in [5,8]:
        Files.extend(readOneFiles(Session, Index, 'M', 'a'))
        Files.extend(readOneFiles(Session, Index, 'M', 'b'))
    save_variable(Files,"/home/zzhang/test/file.data")
    return Files



# Reading files
def readOneFiles(Session, Index, Sex, a):
    wav_path='/home/zzhang/test/IEMOCAP_full_release/Session'+str(Session)+'/dialog/wav/Ses0'+str(Session)+Sex+'_impro0'+str(Index)+a+'.wav'
    EmoEvaluation_path='/home/zzhang/test/IEMOCAP_full_release/Session'+str(Session)+'/dialog/EmoEvaluation/Ses0'+str(Session)+Sex+'_impro0'+str(Index)+a+'.txt'
    f=open(EmoEvaluation_path)
    lines = f.readlines()
    wav, sr = librosa.load(wav_path,sr=16000)

    oneFile=[]
    for line in lines:
        if line.find('[')!=-1:
            if line.find('hap')!=-1:
                oneFile.append(Fragment(line=line.split(), emotion='hap', wav=wav))
            if line.find('neu')!=-1:
                oneFile.append(Fragment(line=line.split(), emotion='neu', wav=wav))
            if line.find('sad')!=-1:
                oneFile.append(Fragment(line=line.split(), emotion='sad', wav=wav))
            if line.find('ang')!=-1:
                oneFile.append(Fragment(line=line.split(), emotion='ang', wav=wav))
            if line.find('fru')!=-1:
                oneFile.append(Fragment(line=line.split(), emotion='fru', wav=wav))
    return oneFile

class Fragment:
    start_time=0
    end_time=0
    spec=np.array([])
    emotion=''
    label=np.array([0,0,0,0,0])
    # hap[1,0,0,0,0]
    # neu[0,1,0,0,0]
    # sad[0,0,1,0,0]
    # ang[0,0,0,1,0]
    # fru[1,0,0,0,1]

    def __init__(self, line, emotion, wav):
        line[0]=list(filter(lambda ch: ch in '0123456789.',line[0]))
        line[0]="".join(line[0])
        start_time=float(line[0])
        line[2]=list(filter(lambda ch: ch in '0123456789.',line[2]))
        line[2]="".join(line[2])
        end_time=float(line[2])
        self.start_name=start_time
        self.end_time=end_time
        # self.fragment=wav[round(start_time*16000):round(end_time*16000)]
        self.spec=np.abs(librosa.stft(wav[round(start_time*16000):round(end_time*16000)], n_fft=800, win_length=640, window='hamming'))
        self.spec=self.spec[0:200,:]
        if emotion=='hap':
            self.label=np.array([1,0,0,0,0])
        if emotion=='neu':
            self.label=np.array([0,1,0,0,0])
        if emotion=='sad':
            self.label=np.array([0,0,1,0,0])
        if emotion=='ang':
            self.label=np.array([0,0,0,1,0])
        if emotion=='fru':
            self.label=np.array([0,0,0,0,1])

class dataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        spec=self.data[index].spec
        spec=spec[np.newaxis, :]
        label=self.data[index].label
        return torch.from_numpy(spec), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

def generatSet(test_size, valid_size):
	Files=load_variable("/home/zzhang/test/file.data")
	random.shuffle(Files)
	test=[]
	valid=[]
	train=[]
	for i in range(0, test_size):
		test.append(Files.pop(random.randint(0,len(Files)-1)))
	for i in range(0, valid_size):
		valid.append(Files.pop(random.randint(0,len(Files)-1)))
	train=Files
	print("Generate Set finish...")
	return train, test, valid

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    spec, label = zip(*data)
    spec_data=[]
    label_data=[]
    for i in label:
        label_data.append(i[np.newaxis, :])
    label=torch.cat(label_data, 0)
    # for i in spec:
    #     i=i.permute([2,1,0])
    for i in spec:
    	spec_data.append(i.permute([2,1,0]))
    spec = torch.nn.utils.rnn.pad_sequence(spec_data, batch_first=True, padding_value=0)
    spec = spec.permute([0, 3, 2, 1])
    return spec, label

# Files=load_variable("/home/zzhang/test/file.data")
# librosa.display.specshow(librosa.amplitude_to_db(Files[0].spec,ref=np.max),y_axis='log', x_axis='time')
# plt.savefig("/home/zzhang/test/pic1.png")
