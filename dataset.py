import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tensorboard




def collate_fn(wqi, data_size):
    wqi = wqi
    len_wqi = len(wqi)
    len_0 = data_size - len_wqi
    if len_wqi != data_size:
        padd = [0 for _ in range(len_0)]
    
    result = padd + wqi
    return result

class KwenDataset(Dataset):
    
    def __init__(self, path, train=True, transform=None, lstm = None):
        self.path = path
        self.img_list = os.listdir(os.path.join(path,'img'))
        self.transform = transform
        self.lstm = lstm
        with open(os.path.join(path, 'label.json'), 'r') as f:
            self.label = json.load(f)
        
        with open(os.path.join(path, 'wqi_score_sorted.json'), 'r') as f:
            self.wqi_score = json.load(f)
        
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, idx):
        # ex:  33.24675_126.571777_500_070925.jpg
        file_name = self.img_list[idx]
        img_path = os.path.join(self.path, 'img',file_name)
        
        label = self.label[file_name]
        
        img = Image.open(img_path)
        img = img.resize((256,192))        
        if self.transform is not None:
            img = self.transform(img)

        if self.lstm:
            lat_loc = self.img_list[idx].split('_')[0]+'_'+self.img_list[idx].split('_')[1]
            
            wqi_vals = list(self.wqi_score[lat_loc].values())
            wqi_key = wqi_vals.index(label)
            wqi_pre = wqi_vals[:wqi_key+1]

            if len(wqi_pre) >=30:
                wqi_pre = wqi_pre[-30:]
            else: pass
            
            wqi = collate_fn(wqi_pre, 32)

            
<<<<<<< HEAD
            return  wqi, img, label
=======
            return  img, wqi, label
>>>>>>> e7ac9e826bf597435d2fdfbbb3eb8f5a95e7db08
        else:
            return img, label