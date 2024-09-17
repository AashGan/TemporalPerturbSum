import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import os
import torch.nn.functional as F
# Adapted from sources : https://github.com/e-apostolidis/PGL-SUM/blob/master/model/data_loader.py,  https://github.com/li-plus/DSNet/blob/master/src/helpers/data_helper.py
datasets = os.listdir('Data/h5datasets')
splits_dir = 'splits'
class VideoData(Dataset):
    def __init__(self,mode,splits_json,split_index = 0,transforms = None,**kwargs):
        super().__init__()
        self.mode = mode
        self.dataset_dict = {}
        self.model_name = kwargs.get('model_name',None)
        splits_json = os.path.join(splits_dir,splits_json)
        print(splits_json)
        assert os.path.exists(splits_json), "The JSON is not configured correctly"
        with open(splits_json) as f:
            data = json.loads(f.read())
            self.all_datapoints = data[split_index][mode +'_keys']
        self._create_data_dict('Data/h5datasets')
        if transforms:
            assert type(transforms) == list, "Ensure the transformations are given as a list"
            self.transforms = transforms
        else:
            self.transforms = False
    def _create_data_dict(self,main_data_path):
        self.all_datasets = [dataset.split('.')[0] for dataset in datasets]
        for dataset in datasets:
            hdf = h5py.File(os.path.join(main_data_path,dataset),'r')
            self.dataset_dict[dataset.split('.')[0]] = hdf # Should give us a dict of all of the data, need to check
    
    def __len__(self):
        return len(self.all_datapoints)
    
    def __getitem__(self,index):
        dataset,video_index = self.all_datapoints[index].split('/')[0],self.all_datapoints[index].split('/')[1]
        features = self.dataset_dict[dataset][video_index]['features'][...]
        shot_bounds = self.dataset_dict[dataset][video_index]['downsampled_shot_boundaries'][...] if 'downsampled_shot_boundaries' in self.dataset_dict[dataset][video_index].keys() else None

        gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
        features = torch.from_numpy(features)
        gtscore = torch.from_numpy(gtscore)
        
        if self.transforms and self.mode=='train':
            for transform in self.transforms:
                if shot_bounds is not None:
                    features,gtscore = transform.shuffle(features,gtscore,shot_bounds = shot_bounds)
                else:
                    features,gtscore = transform.shuffle(features,gtscore)
        
        if self.model_name =='CTV':
            length = len(features)
            if length >=240:
                ids = torch.randperm(length)[:240]
                ids = torch.sort(ids)[0]
            else:
                ids = torch.arange(length).view(1,1,-1).float()
                ids = F.interpolate(ids,size=240, mode='nearest').long().flatten()
            features = features[ids]
            gtscore = gtscore[ids]
        
        if self.mode =='test':
            
            return features,video_index
        
        return features, gtscore
    


class WholeShuffle(object):

    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            perm_indices = torch.randperm(len(features))
            features = features[perm_indices]
            ground_truth = ground_truth[perm_indices]
        return features, ground_truth
    
class Flip(object):
    def __init__(self,**kwargs):

        self.probability = kwargs.get('probability')

    def shuffle(self,features,ground_truth,**kwargs):
        assert len(features) == len(ground_truth), 'Length mismatch'
        if torch.rand((1))>self.probability:
            return features.flip(dims = [0]),ground_truth.flip(dims=[0])
        else:
            return features, ground_truth
    
class Shufflebylength(object):

    def __init__(self,**kwargs):

        self.probability =  kwargs.get('probability')
        self.segment_length =  kwargs.get('segment_length')
    
    def shuffle(self,features,ground_truth,**kwargs):
        if torch.rand(1)> self.probability:
            list_of_segments = [[i,i+self.segment_length] for i in range(0,len(features),self.segment_length)]

            if list_of_segments[len(list_of_segments)-1][1]!= len(features): list_of_segments[len(list_of_segments)-1][1] = len(features)
            shot_order = torch.randperm(len(list_of_segments))
            # Features 
            ground_truth_new = torch.empty_like(ground_truth)

            a_index =0
            for shot in torch.asarray(list_of_segments)[shot_order]:
                ground_truth_new[a_index:a_index+(shot[1]-shot[0]).item()] = ground_truth[shot[0].item():shot[1].item()]
                a_index = a_index +(shot[1]-shot[0]).item()


            features_new = torch.empty_like(features)
            a_index =0
            for shot in torch.asarray(list_of_segments)[shot_order]:
                features_new[a_index:a_index+(shot[1]-shot[0]).item()] = features[shot[0].item():shot[1].item()]
                a_index = a_index +(shot[1]-shot[0]).item()
            return features_new,ground_truth_new
        else:
            return features,ground_truth
    
class SwapOrder(object):

    def __init__(self,**kwargs):

        self.probability = kwargs['probability']
    
    def shuffle(self,features,ground_truth,**kwargs):

        if torch.rand(1)> self.probability:
            half_point = len(features)//2
            features = torch.concat((features[half_point:len(features)],features[0:half_point]),0)
            ground_truth = torch.concat((ground_truth[half_point:len(features)],ground_truth[0:half_point]),0)
        return features,ground_truth


class ChannelShuffle(object):
    def __init__(self,**kwargs) -> None:
        self.probability =  kwargs.get('probability')
        self.groups=  kwargs.get('groups')
    def shuffle(self,features,ground_truth):

        if torch.rand(1)> self.probability:
            features = features.transpose(1,0)
            list_of_segments = [[i,i+self.groups] for i in range(0,len(features),self.groups)]

            if list_of_segments[len(list_of_segments)-1][1]!= len(features): list_of_segments[len(list_of_segments)-1][1] = len(features)
            shot_order = torch.randperm(len(list_of_segments))

            features_new = torch.empty_like(features)
            a_index =0
            for shot in torch.asarray(list_of_segments)[shot_order]:
                features_new[a_index:a_index+(shot[1]-shot[0]).item()] = features[shot[0].item():shot[1].item()]
                a_index = a_index +(shot[1]-shot[0]).item()
            return features_new.transpose(1,0),ground_truth
        else:
            return features,ground_truth
        



class ShuffleShots(object):
    def __init__(self,**kwargs):
        self.probability = kwargs.get('probability')
    def shuffle(self,features,ground_truth,shot_bounds):
        if torch.rand(1)>self.probability:
            shuffled_shots = torch.randperm(len(shot_bounds))
            features_new = torch.empty_like(features)
            ground_truth_new = torch.empty_like(ground_truth)
            a_index = 0
            for shot in torch.asarray(shot_bounds)[shuffled_shots]:
                features_new[shot[0].item():1+shot[1].item()] = features[a_index:1+a_index+(shot[1]-shot[0]).item()] 
                ground_truth_new[shot[0].item():1+shot[1].item()] = ground_truth[a_index:1+a_index+(shot[1]-shot[0]).item()] 
                a_index = 1+a_index +(shot[1]-shot[0]).item()
            return features_new,ground_truth_new
        return features,ground_truth
    
class ShuffleNeighbourShots(object):
    def __init__(self,**kwargs):
        self.probability = kwargs.get('probability')
    def shuffle(self,features,ground_truth,shot_bounds):
        if torch.rand(1)>self.probability:
            shot_shuffle_lengths = np.arange(0,len(shot_bounds)+1,3)
            if len(shot_bounds)%3!=0:
                shot_shuffle_lengths = np.append(shot_shuffle_lengths,len(shot_bounds))
            shuffled_shots = torch.cat([torch.arange(shot_shuffle_lengths[j],shot_shuffle_lengths[j+1])[torch.randperm(len(torch.arange(shot_shuffle_lengths[j],shot_shuffle_lengths[j+1]))) ] for j in range(len(shot_shuffle_lengths)-1)],dim=0)
            features_new = torch.empty_like(features)
            ground_truth_new = torch.empty_like(ground_truth)
            a_index = 0
            for shot in torch.asarray(shot_bounds)[shuffled_shots]:
                features_new[shot[0].item():1+shot[1].item()] = features[a_index:1+a_index+(shot[1]-shot[0]).item()] 
                ground_truth_new[shot[0].item():1+shot[1].item()] = ground_truth[a_index:1+a_index+(shot[1]-shot[0]).item()] 
                a_index = 1+a_index +(shot[1]-shot[0]).item()
            return features_new,ground_truth_new
        return features,ground_truth
    
class IntraShotShuffle(object):
    def __init__(self,**kwargs):
        self.probability = kwargs.get('probability')
    
    def shuffle(self,features,gtscore,**kwargs):
        if torch.rand(1)>self.probability:
            shot_boundaries = kwargs.get('shot_bounds')
            for shot_bound in torch.asarray(shot_boundaries):
                randperm = torch.randperm(len(torch.arange(shot_bound[0],shot_bound[1]+1)))
                features[shot_bound[0]:shot_bound[1]+1] = features[shot_bound[0]:shot_bound[1]+1][randperm]
                gtscore[shot_bound[0]:shot_bound[1]+1] = gtscore[shot_bound[0]:shot_bound[1]+1][randperm]
            return features,gtscore
        return features,gtscore
    


class MLPDataloader(Dataset):
    def __init__(self,mode,splits_json,split_index = 0,transforms = None,**kwargs) -> None:
        super().__init__()
        self.mode = mode
        self.dataset_dict = {}
        splits_json = os.path.join(splits_dir,splits_json)
        assert os.path.exists(splits_json), "The JSON is not configured correctly"
        with open(splits_json) as f:
            data = json.loads(f.read())
            self.all_datapoints = data[split_index][mode +'_keys']
        self._create_data_dict('Data\\h5datasets')
        if transforms:
            assert type(transforms) == list, "Ensure the transformations are given as a list"
            self.transforms = transforms
        else:
            self.transforms = False
        self.all_feat_img_pairs = self.create_all_pairs()
    
    def create_all_pairs(self):
        features_matrix = []
        gt_score_matrix = []
        for i in range(len(self.all_datapoints)):
            dataset,video_index = self.all_datapoints[i].split('/')[0],self.all_datapoints[i].split('/')[1]
            features = self.dataset_dict[dataset][video_index]['features'][...]
            gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
            gtscore-=gtscore.min()
            gtscore/=gtscore.max()
            features_matrix.append(features)
            gt_score_matrix.append(gtscore)
        return [np.concatenate(features_matrix,axis=0),np.concatenate(gt_score_matrix,axis=0)]

    def _create_data_dict(self,main_data_path):
        self.all_datasets = [dataset.split('.')[0] for dataset in datasets]
        for dataset in datasets:
            hdf = h5py.File(os.path.join(main_data_path,dataset),'r')
            self.dataset_dict[dataset.split('.')[0]] = hdf # Should give us a dict of all of the data, need to check
    def __getitem__(self,index):
        if self.mode =='test':
            dataset,video_index = self.all_datapoints[index].split('/')[0],self.all_datapoints[index].split('/')[1]
            features = self.dataset_dict[dataset][video_index]['features'][...]
            return features,video_index

            
        features = torch.from_numpy(self.all_feat_img_pairs[0][index])
        gtscore = torch.tensor(self.all_feat_img_pairs[1][index])
        return features,gtscore
    def __len__(self):
        if self.mode=='train':
            return len(self.all_feat_img_pairs[0])
        else:
            return len(self.all_datapoints)
        
shuffle_dict  = {'WholeShuffle': WholeShuffle,'SwapOrder':SwapOrder,'Shufflebylength':Shufflebylength,'Flip':Flip,'ChannelShuffle':ChannelShuffle,'IntraShotShuffle':IntraShotShuffle,'ShuffleNeighbourShots':ShuffleNeighbourShots,'ShuffleShots':ShuffleShots}