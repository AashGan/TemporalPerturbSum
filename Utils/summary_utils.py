# Taken from PGLSUM github
#TODO sanity check
import numpy as np
from scipy.stats import rankdata
from scipy.stats import kendalltau, spearmanr
import hdf5storage
from .dicts import tvsum_video_dict,summe_video_dict
def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]  # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]
        # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

        # Select the best shots using the knapsack implementation
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)
        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries


def generate_summary_single(shot_bound,score,n_frames,positions,return_shot_info = False):
    frame_init_scores = score
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * 0.15)

    selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1
    if return_shot_info:
        return shot_lengths, shot_imp_scores,selected,summary
    return summary


def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        
        item_dict = {
        'video': video[0, 0],
        'category': category[0, 0],
        'title': title[0, 0],
        'length': length[0, 0],
        'nframes': nframes[0, 0],
        'user_anno': user_anno,
        'gt_score': gt_score
        }
        
        data_list.append((item_dict))
    
    return data_list
#TODO  : Mention it and experiment with both upsampling the model output or downsampling the user score.
def evaluate_correlation(scores,dataset,video_names,dataset_name='tvsum'):
    if dataset_name=="tvsum":
        data = load_tvsum_mat('Utils//ydata-tvsum50.mat')
        all_correlations_tau_split =[] 
        all_correlations_spearman_split =[]
        evaluation_dict = {}
        for score,video_name in zip(scores,video_names):
            video_number = int(video_name.split('_')[1])
            all_user_summary = data[video_number-1]['user_anno'].T
            all_correlations_tau = []
            all_correlations_spearman = []
            evaluation_dict[video_name] = {}
            i=0 
            for user_summary in all_user_summary:
                down_sampled_summary = (user_summary/user_summary.max())[::15]
                correlation_tau = kendalltau(-rankdata(down_sampled_summary),-rankdata(score))[0]
                correlation_spear = spearmanr(down_sampled_summary,score)[0]

                all_correlations_tau.append(correlation_tau)
                all_correlations_spearman.append(correlation_spear)
            #print(f"The Kendall and Spearman coefficent for {video_name} : {np.mean(all_correlations_tau)} , {np.mean(all_correlations_spearman)}")
            evaluation_dict[video_name]['kendall'] = np.mean(all_correlations_tau)
            evaluation_dict[video_name]['spearman'] = np.mean(all_correlations_spearman)
            all_correlations_tau_split.append(np.mean(all_correlations_tau))
            all_correlations_spearman_split.append(np.mean(all_correlations_spearman))
        #print(f"Overall Split Kendall Split: {np.mean(all_correlations_tau_split)} ")
        #print(f"Overall Split Spearman Split: {np.mean(all_correlations_spearman_split)} ")
        evaluation_dict['Average_Kendall'] = np.mean(all_correlations_tau_split)
        evaluation_dict['Average_Spearman'] = np.mean(all_correlations_spearman_split)
        return evaluation_dict
    elif dataset_name == "summe":
        evaluation_dict = {}
        all_correlations_tau_split =[] 
        all_correlations_spearman_split =[]
        for score,video_name in zip(scores,video_names):
            evaluation_dict[video_name] = {}
            user_summarie = dataset[video_name]['user_summary']
            averaged_downsampled_summary = np.average(user_summarie,axis=0)[::15]
            kendall_score = kendalltau(rankdata(averaged_downsampled_summary),rankdata(score))[0]
            spearman_score = spearmanr(averaged_downsampled_summary,score)[0]
            #print(f"The kendall and spear man coefficent for video {video_name} : {kendall_score} , {spearman_score}")
            evaluation_dict[video_name]['kendall'] = kendall_score
            evaluation_dict[video_name]['spearman'] = spearman_score
            all_correlations_tau_split.append(kendall_score)
            all_correlations_spearman_split.append(spearman_score)
        #print(f"Overall Split Kendall Split: {np.mean(all_correlations_tau_split)} ")
        #print(f"Overall Split Spearman Split: {np.mean(all_correlations_spearman_split)} ")
        evaluation_dict['Average_Kendall'] = np.mean(all_correlations_tau_split)
        evaluation_dict['Average_Spearman'] = np.mean(all_correlations_spearman_split)
        return evaluation_dict
    else:
         print("Dataset incorrect")

    return {}

def evaluate_f1score(outputs,dataset,names,data_name = 'summe'):
    eval_metric = 'avg' if data_name == 'tvsum' else 'max'
    fms = []
    all_user_summary, all_shot_bound, all_nframes, all_positions,all_scores = [], [], [], [] ,[]
    for name  in names:
        cps = dataset[name]['change_points'][...]
        num_frames = dataset[name]['n_frames'][...]
        nfps = dataset[name]['n_frame_per_seg'][...].tolist()
        positions = dataset[name]['picks'][...]
        user_summary = dataset[name]['user_summary'][...]
        all_user_summary.append(user_summary)
        all_shot_bound.append(cps)
        all_nframes.append(num_frames)
        all_positions.append(positions)

    machine_summaries = generate_summary(all_shot_bound, outputs, all_nframes, all_positions)
    for machine_summary,user_summary,name in zip(machine_summaries,all_user_summary,names):
        fm= evaluate_summary(machine_summary, user_summary, eval_metric)
        #print(f"{name[0]} F1 : {fm}")
        fms.append(fm)
        return fms              
     
      
def knapSack(W, wt, val, n):
	""" Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
	no concept of putting some part of item in the knapsack.

	:param int W: Maximum capacity -in frames- of the knapsack.
	:param list[int] wt: The weights (lengths -in frames-) of each video shot.
	:param list[float] val: The values (importance scores) of each video shot.
	:param int n: The number of the shots.
	:return: A list containing the indices of the selected shots.
	"""
	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i - 1] <= w:
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
			else:
				K[i][w] = K[i - 1][w]

	selected = []
	w = W
	for i in range(n, 0, -1):
		if K[i][w] != K[i - 1][w]:
			selected.insert(0, i - 1)
			w -= wt[i - 1]

	return selected

def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)

def eval_summary(outputs,dataset,names,data_name='summe'):
    eval_metric = 'avg' if data_name == 'tvsum' else 'max'
    fms = []
    all_user_summary, all_shot_bound, all_nframes, all_positions,all_scores = [], [], [], [] ,[]
    for name  in names:
        cps = dataset[name]['change_points'][...]
        num_frames = dataset[name]['n_frames'][...]
        nfps = dataset[name]['n_frame_per_seg'][...].tolist()
        positions = dataset[name]['picks'][...]
        user_summary = dataset[name]['user_summary'][...]
        all_user_summary.append(user_summary)
        all_shot_bound.append(cps)
        all_nframes.append(num_frames)
        all_positions.append(positions)

    machine_summaries = generate_summary(all_shot_bound, outputs, all_nframes, all_positions)
    for machine_summary,user_summary,name in zip(machine_summaries,all_user_summary,names):
        fm= evaluate_summary(machine_summary, user_summary, eval_metric)
        #print(f"{name[0]} F1 : {fm}")
        fms.append(fm)
    return fms

def generate_f1_results(outputs,dataset,names,data_name='summe'):
    all_f1_scores = eval_summary(outputs,dataset,names,data_name)
    result_dict = {}
    for i in range(len(outputs)):
        result_dict[names[i]] = all_f1_scores[i]
    result_dict['Average F1'] = np.mean(all_f1_scores)
    return result_dict
        
          
     
def compute_average_results(f1_score_dict,correlation_dict):
     # First iterate oer f1, second take the correlation for each

    average_f1_overall = np.mean([f1_score_dict[name] for name in list(f1_score_dict.keys())])
    f1_score_dict['Overall Performance'] = average_f1_overall

    # Do this for kendall and Spearman

    average_kendall = np.mean([correlation_dict[name]['Kendall'] for name in list(correlation_dict.keys())])
    
    average_spearman = np.mean([correlation_dict[name]['Spearman'] for name in list(correlation_dict.keys())])
    correlation_dict['Overall Spearman'] = average_spearman
    correlation_dict['Overall Kendall'] = average_kendall
    return f1_score_dict,correlation_dict



def change_key_names(F1_dict,Correlation_dict,dataset):
    if dataset == 'tvsum':
        F1_dict = {tvsum_video_dict[key]: F1_dict[key] for key in list(F1_dict.keys())}
        Correlation_dict = {tvsum_video_dict[key]: Correlation_dict[key] for key in list(Correlation_dict.keys())}
    elif dataset == 'summe':
        F1_dict = {summe_video_dict[key]: F1_dict[key] for key in list(F1_dict.keys())}
        Correlation_dict = {summe_video_dict[key]: Correlation_dict[key] for key in list(Correlation_dict.keys())}
    else:
        print('Wrong Dataset')
    return F1_dict,Correlation_dict