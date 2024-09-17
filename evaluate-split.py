import torch
from Model import model_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from Utils import *
from Data import VideoData,shuffle_dict
import sys
import numpy as np
import argparse
import h5py

## add cwd to path if it isn't in the path already



torch.manual_seed(34523421)
results_dir = "Results"
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

parser = argparse.ArgumentParser(description = "Running models over a split")
parser.add_argument('--config_path',type= str,required=True,help = "Path to the config file for the train run")
parser.add_argument('--weights_path',type = str,default = "weights",help = 'Path to load the models')
parser.add_argument('--delete_weights',action='store_true' ,help = 'Delete the weights')




def inference():

    args = parser.parse_args()
    with open(args.config_path,'r') as config_file:
        config = json.load(config_file)
    if args.delete_weights==True:
        print("IMPORTANT NOTE: WEIGHTS ARE DELETED, RERUN TRAINING TO GET THEM BACK")
    assert config['Model_params']['Model'] in model_dict.keys(), "Model is not available, modify dictionary to include them or check spelling"
    

    dataset_name = config['split'].split("_")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    modelclass = model_dict[config['Model_params']['Model']]
    
    if "params" in config['Model_params']:
        params = config['Model_params']['params']
    else:
        params = {}    # Running checks and creating results for each experiment
        #Save it as results/experiment/model
    
    #---------------------------------------------------- BEST Correlation runs-------------------------------------------------------------------------
    
    if not os.path.exists(os.path.join(results_dir,config['save_name'],'best_corr',dataset_name,config['Model_params']['Model'])):
        os.makedirs(os.path.join(results_dir,config['save_name'],'best_corr',dataset_name,config['Model_params']['Model']))
    result_dir = os.path.join(results_dir,config['save_name'],'best_corr',dataset_name,config['Model_params']['Model'])
    if not os.path.exists(os.path.join(result_dir,'F1')):
        os.mkdir(os.path.join(result_dir,'F1'))
    if not os.path.exists(os.path.join(result_dir,'Correlation')):
        os.mkdir(os.path.join(result_dir,'Correlation'))
    if not os.path.exists(os.path.join(result_dir,'Output')):
        os.mkdir(os.path.join(result_dir,'Output'))
    result_f1_dir = os.path.join(result_dir,'F1')
    result_corr_dir = os.path.join(result_dir,'Correlation')
    result_f1_json = os.path.join(result_dir,'F1','results.json')
    result_corr_json = os.path.join(result_dir,'Correlation','results.json')
    result_out_json = os.path.join(result_dir,'Output','outputs.json')



    splits = config['total_splits']


    weight_path = os.path.join(args.weights_path,config['save_name'],dataset_name,config['Model_params']['Model'])
    assert os.path.exists(os.path.join(args.weights_path,config['save_name'],dataset_name,config['Model_params']['Model'])), "Model weights do not exist or pathing is incorrect, check path"
    dataset = h5py.File(os.path.join('Data/h5datasets',config['datapath']+'.h5'))
    
    all_splits_f1_scores = {}
    all_splits_correlations = {}
    output_dict ={}
    for split in range(splits):
        model = modelclass(**params)
        testdata = VideoData('test',config['split'],split)
        testloader = DataLoader(testdata,batch_size=1,shuffle=False)
        weight_path_split = os.path.join(weight_path,f"split_{split+1}",'best_run_corr.pth')
        model.load_state_dict(torch.load(weight_path_split,map_location=device))
        name_list = []
        output_list = []
        model.eval()
        model.to(device)
        for inputs,names in tqdm(testloader,ncols=100):
            inputs = inputs.to(device)
            with torch.no_grad():
                importance_scores = model(inputs)
                if len(importance_scores.shape)>2:
                    importance_scores = importance_scores.squeeze(-1)
            importance_scores = importance_scores[0].to('cpu').tolist()
            output_list.append(importance_scores)
            name_list.append(names[0])
            output_dict[str(names[0])] = importance_scores
        result_f1_dict = generate_f1_results(output_list,dataset,name_list,dataset_name) # This needs to be dumped into a JSON for each split
        correlation_dict = evaluate_correlation(output_list,dataset,name_list,dataset_name) # This as well
        

        split_save_f1_name = os.path.join(result_f1_dir,f"split_{split+1}.json")
        split_save_corr_name = os.path.join(result_corr_dir,f"split_{split+1}.json")
        split_save_f1_noname = os.path.join(result_f1_dir,f"split_{split+1}_noname.json")
        split_save_corr_noname = os.path.join(result_corr_dir,f"split_{split+1}_noname.json")
        
        
        all_splits_f1_scores[f'split_{split+1}'] = result_f1_dict['Average F1']
        all_splits_correlations[f'split_{split+1}'] = {}
        all_splits_correlations[f'split_{split+1}']['Kendall']= correlation_dict['Average_Kendall']
        all_splits_correlations[f'split_{split+1}']['Spearman']= correlation_dict['Average_Spearman']
       
        # Pop the average keys and then use it to save the split with name of the original
        result_f1_dict.pop('Average F1')
        correlation_dict.pop('Average_Kendall')
        correlation_dict.pop('Average_Spearman')
        with open(split_save_f1_noname,'w') as json_file:
            json.dump(result_f1_dict,json_file,indent=4)
        with open(split_save_corr_noname,'w') as json_file:
            json.dump(correlation_dict,json_file,indent=4)
        result_f1_dict,correlation_dict = change_key_names(result_f1_dict,correlation_dict,dataset_name)

        with open(split_save_f1_name,'w') as json_file:
            json.dump(result_f1_dict,json_file,indent=4)
        with open(split_save_corr_name,'w') as json_file:
            json.dump(correlation_dict,json_file,indent=4)
        
        if args.delete_weights==True:
            os.remove(os.path.join(weight_path,f"split_{split+1}",'best_run_corr.pth'))

        

    print(all_splits_correlations)
    result_f1_dict_final,correlation_dict_final = compute_average_results(all_splits_f1_scores,all_splits_correlations)


    with open(result_f1_json,'w') as json_file:
        json.dump(result_f1_dict_final,json_file,indent = 4 )
    with open(result_corr_json,'w') as json_file:
        json.dump(correlation_dict_final,json_file,indent = 4 )
    with open(result_out_json,'w') as json_file:
        json.dump(output_dict,json_file,indent = 4 )
   

#------------------------------------------Best F1 score runs---------------------------------------------------------
    if not os.path.exists(os.path.join(results_dir,config['save_name'],'best_f1',dataset_name,config['Model_params']['Model'])):
        os.makedirs(os.path.join(results_dir,config['save_name'],'best_f1',dataset_name,config['Model_params']['Model']))
    result_dir = os.path.join(results_dir,config['save_name'],'best_f1',dataset_name,config['Model_params']['Model'])
    # Check if the F1 score and the correlation directories for the results exist
    if not os.path.exists(os.path.join(result_dir,'F1')):
        os.mkdir(os.path.join(result_dir,'F1'))
    if not os.path.exists(os.path.join(result_dir,'Correlation')):
        os.mkdir(os.path.join(result_dir,'Correlation'))
    if not os.path.exists(os.path.join(result_dir,'Output')):
        os.mkdir(os.path.join(result_dir,'Output'))
    result_f1_dir = os.path.join(result_dir,'F1')
    result_corr_dir = os.path.join(result_dir,'Correlation')
    result_f1_json = os.path.join(result_dir,'F1','results.json')
    result_corr_json = os.path.join(result_dir,'Correlation','results.json')
    result_out_json = os.path.join(result_dir,'Output','outputs.json')



    splits = config['total_splits']
    # Loading the weights:
        # Loading the weights as : "weights/experiment/model"
    weight_path = os.path.join(args.weights_path,config['save_name'],dataset_name,config['Model_params']['Model'])
    assert os.path.exists(os.path.join(args.weights_path,config['save_name'],dataset_name,config['Model_params']['Model'])), "Model weights do not exist or pathing is incorrect, check path"
    dataset = h5py.File(os.path.join('Data/h5datasets',config['datapath']+'.h5'))
    
    output_dict ={}
    all_splits_f1_scores = {}
    all_splits_correlations = {}
    for split in range(splits):
        model = modelclass(**params)
        testdata = VideoData('test',config['split'],split)
        testloader = DataLoader(testdata,batch_size=1,shuffle=False)
        weight_path_split = os.path.join(weight_path,f"split_{split+1}",'best_run_f1.pth')
        model.load_state_dict(torch.load(weight_path_split,map_location=device))
        
        name_list = []
        output_list = []
        model.eval()
        model.to(device)
        for inputs,names in tqdm(testloader,ncols=100):
            
            inputs = inputs.to(device)
            with torch.no_grad():
                importance_scores = model(inputs)
                if len(importance_scores.shape)>2:
                    importance_scores = importance_scores.squeeze(-1)
            importance_scores = importance_scores[0].to('cpu').tolist()
            output_list.append(importance_scores)
            name_list.append(names[0])
            output_dict[str(names[0])] = importance_scores
        
        result_f1_dict = generate_f1_results(output_list,dataset,name_list,dataset_name) # This needs to be dumped into a JSON for each split
        correlation_dict = evaluate_correlation(output_list,dataset,name_list,dataset_name) # This as well
        
        # Saving the split in the respective directories

        split_save_f1_name = os.path.join(result_f1_dir,f"split_{split+1}.json")
        split_save_corr_name = os.path.join(result_corr_dir,f"split_{split+1}.json")
        split_save_f1_noname = os.path.join(result_f1_dir,f"split_{split+1}_noname.json")
        split_save_corr_noname = os.path.join(result_corr_dir,f"split_{split+1}_noname.json")
        
        
        all_splits_f1_scores[f'split_{split+1}'] = result_f1_dict['Average F1']
        all_splits_correlations[f'split_{split+1}'] = {}
        all_splits_correlations[f'split_{split+1}']['Kendall']= correlation_dict['Average_Kendall']
        all_splits_correlations[f'split_{split+1}']['Spearman']= correlation_dict['Average_Spearman']
       
        # Pop the average keys and then use it to save the split with name of the original

        result_f1_dict.pop('Average F1')
        correlation_dict.pop('Average_Kendall')
        correlation_dict.pop('Average_Spearman')

        with open(split_save_f1_noname,'w') as json_file:
            json.dump(result_f1_dict,json_file,indent=4)
        with open(split_save_corr_noname,'w') as json_file:
            json.dump(correlation_dict,json_file,indent=4)
        result_f1_dict,correlation_dict = change_key_names(result_f1_dict,correlation_dict,dataset_name)

        with open(split_save_f1_name,'w') as json_file:
            json.dump(result_f1_dict,json_file,indent=4)
        with open(split_save_corr_name,'w') as json_file:
            json.dump(correlation_dict,json_file,indent=4)
        if args.delete_weights==True:
            os.remove(os.path.join(weight_path,f"split_{split+1}",'best_run_f1.pth'))

        

    print(all_splits_correlations)
    result_f1_dict_final,correlation_dict_final = compute_average_results(all_splits_f1_scores,all_splits_correlations)


    with open(result_f1_json,'w') as json_file:
        json.dump(result_f1_dict_final,json_file,indent = 4 )
    with open(result_corr_json,'w') as json_file:
        json.dump(correlation_dict_final,json_file,indent = 4 )
        
    with open(result_out_json,'w') as json_file:
        json.dump(output_dict,json_file,indent = 4 )
   






if __name__ == "__main__":
    inference()
