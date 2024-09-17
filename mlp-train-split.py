import torch
from Model import model_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from Utils import eval_summary, evaluate_correlation,loss_dict
import torch.optim as optim
from Data import VideoData,shuffle_dict,MLPDataloader
import sys
import numpy as np
import torch.nn.init as init
import argparse
import h5py
## TODO: Check on the optimization/scheduler
## add cwd to path if it isn't in the path already
torch.manual_seed(34523421)
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

# Take arguments to save weights and to save tensorboard
    
parser = argparse.ArgumentParser(description = "Running train models over a split")
parser.add_argument('--config_path',type= str,required=True,help = "Path to the config file for the train run")
parser.add_argument('--Tensorboard_dir',type=str,default ="tensorboard_runs",help= 'Directory towards the tensorboard')
parser.add_argument('--save_path',type = str,default = "weights",help = 'Path to save the models')



## Helper functions for training stability

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.kaiming_uniform_(m.weight)
        #init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def train():
    args = parser.parse_args()
    with open(args.config_path,'r') as config_file:
        config = json.load(config_file)
    
    assert config['Model_params']['Model'] in ['MLPM','Attention'], "This file is only to train the Multi-layer perceptron or Self Attention"
    #TODO perhaps make this a bit more flexible with the names

    dataset_name = config['split'].split("_")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelclass = model_dict[config['Model_params']['Model']]


    criterion = loss_dict[config['loss_function']]()
    num_epochs = config["num_epochs"]
    # Running the checks to see if certain filenames have been created so that we don't have to do them manually

        # Format for saving: Experiment/dataset/model
    if not os.path.exists(os.path.join(args.save_path,config['save_name'],dataset_name,config['Model_params']['Model'] )):
        os.makedirs(os.path.join(args.save_path,config['save_name'],dataset_name,config['Model_params']['Model'] ))
    
    save_path = os.path.join(args.save_path,config['save_name'],dataset_name,config['Model_params']['Model'] )
    print(save_path)
    dataset = h5py.File(os.path.join('Data/h5datasets',config['datapath']+'.h5'))
    if "params" in config['Model_params']:
        params = config['Model_params']['params']
    else:
        params = {}

    if config['data_aug'] :
        data_augmentations  = [shuffle_dict[data_aug](**config['data_aug'][data_aug]) for data_aug in config['data_aug']]
    else:
        data_augmentations = []
    # Since its a run over all split, it will loop over them.
    splits = config['total_splits']
    
    for split in range(splits):
        model_name = config['Model_params']['Model']
        print(f"Running Split:  {split+1}  for model: {config['Model_params']['Model']}")
        model = modelclass(**params)
        batchloader = MLPDataloader('train',config['split'],split,transforms=data_augmentations)
        batchloader = DataLoader(batchloader,batch_size=config['batch'],shuffle=True)
        testdata = VideoData('test',config['split'],split)
        testloader = DataLoader(testdata,batch_size=1,shuffle=False)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],weight_decay=config['reg'])
        best_f1_score = -float('inf')
        best_correlation = -float('inf')
        model.to(device)
        if 'gradnorm_clip' in config:
            gradnorm_clip = config['gradnorm_clip']
        else:
            gradnorm_clip = 5
        # Make the directory for the split if it doesn't exist 
        if not os.path.exists(os.path.join(save_path,f'split_{split+1}')):
            os.mkdir(os.path.join(save_path,f'split_{split+1}'))
        save_path_split = os.path.join(save_path,f'split_{split+1}')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for data in tqdm(batchloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if len(outputs.shape)>2:
                    outputs = outputs.squeeze(-1)
                
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm_clip)
                optimizer.step()
                running_loss += loss.item()
                total_samples+=1
            epoch_loss = running_loss / len(batchloader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

            model.eval()
            test_datapoints = []
            test_names = []
            
            #TODO: Check might be better to use F1 orw correlation as the score comparison for model saving.

            print(f"Compute F1 and Correlation for epoch: {epoch+1}")
            for inputs_t,names in tqdm(testloader,ncols=len(testdata)):
                with torch.no_grad():
                    importance_scores = model(inputs_t.to(device))

                importance_scores = importance_scores[0].to('cpu').tolist()
                test_datapoints.append(importance_scores)
                test_names.append(names[0])
            all_scores = eval_summary(test_datapoints,dataset,test_names,dataset_name)
    
            correlation_dict = evaluate_correlation(test_datapoints ,dataset,test_names,dataset_name)
            
            if correlation_dict['Average_Kendall']> best_correlation:    
                print(f"Saving epoch {epoch+1}")
                best_correlation = correlation_dict['Average_Kendall']
                print(f"Best Correlation Score:  {epoch+1}: {correlation_dict['Average_Kendall']} ")
                torch.save(model.state_dict(), os.path.join(save_path_split,"best_run_corr" + ".pth"))

            if np.mean(all_scores).item() > best_f1_score:
                best_f1_score = np.mean(all_scores).item()
                print(f"Best F1 Score:  {epoch+1}: {best_f1_score} ")
                torch.save(model.state_dict(), os.path.join(save_path_split,"best_run_f1" + ".pth"))

        print(f'Best F1 score for split {split+1}: {best_f1_score} ')
        print(f'Best Correlation for split {split+1}: {best_correlation} ')
    print('Completed Training')
    

if __name__ == "__main__":
    train()   
