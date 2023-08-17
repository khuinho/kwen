import pandas as pd
import math
import os
from numba import jit


class nsf_wqi:

    
    def __init__(self, data_path = 'calculate' ):
        self.q_data = pd.read_csv(os.path.join(data_path, 'Q_calculate.csv'))
        self.temp_solubility_data = pd.read_csv(os.path.join(data_path, 'temp_solubility_calculate.csv')).dropna()
        self.weight_data = pd.read_csv(os.path.join(data_path, 'weight.csv'))
        
    def calc_qi(self,item, input_value):
        #try:
            q_data_item = self.q_data[[item, item+'QI']].dropna()
            water_item = dict(q_data_item[item])
            water_item_qi = dict(q_data_item[item+'QI'])
            
            input_value = input_value
            first = False
            last = False
            
            for idx, val in water_item.items():
                if input_value <= water_item[0]:
                    find_idx = 0
                    first = True
                    break
                if idx == len(water_item)-1:
                   find_idx = idx
                   last = True
                   continue
               
                if input_value >= val and input_value <water_item[idx+1]:
                    find_idx = idx
                    break
                
            if first:
                qi_value = water_item_qi[0]
            elif last:
                qi_value = water_item_qi[len(water_item)-1]
            else:
                w1 = abs(input_value - water_item[find_idx+1])/(water_item[find_idx+1]-water_item[find_idx])
                w2 = abs(input_value - water_item[find_idx])/(water_item[find_idx+1]-water_item[find_idx])
                qi_value = w1*water_item_qi[find_idx] + w2*water_item_qi[find_idx+1]
            
            return qi_value
        #except:
        #    print('No item in data item must in {}'.format(str(['BOD','Ecoli','Nitrate','pH','TempChange','TDS','Phosphate','Turbidity','DO_percent'])))
            
    def calc_solubility(self, temp, DO):
        solubility_data = dict(self.temp_solubility_data)
        temp = round(temp)
        solubility = solubility_data['Solubility'][temp]
        DO_percent = DO/solubility
        return DO_percent*100
        
    def calc_weight(self, item_list):
        weight_data = dict(self.weight_data)
        value_sum = 0
        weight_dict = {}
        
        
        for item, value in zip(weight_data['item'],weight_data['value']):
            if item in item_list:
                value_sum += value
                weight_dict[item] = value
        
        for item in weight_dict:
            weight_dict[item] = round(weight_dict[item]/value_sum, 4)
        
        return weight_dict
    
    def calc_wqi(self, wqs_result, print_detail = False):
        
        nsf_wqi_result = {'q':{}, 'w':{}}
        items = list(wqs_result.keys())
        if 'DO' in items and 'Temp' in items:
            DO_percent = self.calc_solubility(wqs_result['Temp'],wqs_result['DO'])    
            del wqs_result['DO']
            del wqs_result['Temp']
            wqs_result['DO_percent'] = DO_percent
        
        items = list(wqs_result.keys())
        weight = self.calc_weight(items)
        nsfw_score = 0
        
        for item in weight.keys():
            nsf_wqi_result['w'][item] = weight[item] 
            nsf_wqi_result['q'][item] = self.calc_qi(item, wqs_result[item])
            #print(item, weight, nsf_wqi_result['w'][item],nsf_wqi_result['q'][item])
            nsfw_score += round(nsf_wqi_result['w'][item]*nsf_wqi_result['q'][item],2)
        
        if print_detail:
            print(nsf_wqi_result)
            print(nsfw_score)
        
        return nsfw_score 