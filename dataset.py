import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader





def collate_fn(wqi, data_size):
    wqi = wqi
    len_wqi = len(wqi)
    len_0 = data_size - len_wqi
    if len_wqi != data_size:
        padd = [0 for _ in range(len_0)]
        result = padd + wqi
        result = result[data_size*-1:]
        return result
    else:
        return wqi

class KwenDataset(Dataset):
    
    def __init__(self, path, len_wqi, train=True, transform=None, lstm = None, label_type = True):
        self.path = path
        self.img_list = os.listdir(os.path.join(path,'img_resize'))
        self.transform = transform
        self.lstm = lstm
        self.label_type = label_type
        
        if label_type == True:
            with open(os.path.join(path, 'label.json'), 'r') as f:
                self.label = json.load(f)
        else:
            with open(os.path.join(path, 'convert_label.json'), 'r') as f:
                self.label = json.load(f)
                    
        if label_type == True:
            with open(os.path.join(path, 'wqi_score_sorted.json'), 'r') as f:
                self.wqi_score = json.load(f)
        else:
            with open(os.path.join(path, 'wqi_score_convert.json'), 'r') as f:
                self.wqi_score = json.load(f)
        
        self.len_wqi = len_wqi
        
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, idx):
        # ex:  33.24675_126.571777_500_070925.jpg
        file_name = self.img_list[idx]
        img_path = os.path.join(self.path, 'img_resize',file_name)
        
        label = self.label[file_name]
        
        img = Image.open(img_path)
        img = img.resize((256,192))        
        if self.transform is not None:
            img = self.transform(img)

        if self.lstm:
            lat_loc = self.img_list[idx].split('_')[0]+'_'+self.img_list[idx].split('_')[1]
            
            d = file_name.split('_')[-1].split('.')[0]
            idx_d = 0
            while True:
                idx_d += 1
                try:
                    wqi_key = list(self.wqi_score[lat_loc].keys()).index(d)
                    break
                except:
                    if idx_d%2 == 1:
                        d = int(d)
                        d +=idx_d
                        d= str(d)
                    else:
                        d = int(d)
                        d -=idx_d
                        d= str(d)
                    
            wqi_vals = list(self.wqi_score[lat_loc].values())
            
            #wqi_key = wqi_vals.index(label)

            wqi_pre = wqi_vals[:wqi_key+1]
                
            if len(wqi_pre) >=self.len_wqi:
                wqi_pre = wqi_pre[-1 * self.len_wqi:]
                
            else: pass
            
            wqi = collate_fn(wqi_pre, self.len_wqi)

            
            return   img,wqi, label 
        else:
            return img, label