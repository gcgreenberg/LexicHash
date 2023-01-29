from os.path import join
import numpy as np
import importlib
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt

PROJ_DIR = '/home/gcgreen2/alignment'
# GT_DIR = join(PROJ_DIR, 'spectral_jaccard_similarity/groundTruths')

def get_gt_df(gt_path, seq_lens, min_theta=0.3):
    gt_df = pd.read_csv(gt_path, sep='\t', header=None, names=['i1','i2','overlap','direc'],
                       dtype={'i1':int,'i2':int,'overlap':float,'direc':str})
    thetas = get_thetas(gt_df, seq_lens)
    gt_df = gt_df[thetas > min_theta]
    return gt_df

def get_gt_df_nctc(gt_path, seq_lens, min_theta=0.3):
    gt_df = pd.read_csv(gt_path, sep='\t', header=None, names=['i1','i2','overlap'],
                       dtype={'i1':int,'i2':int,'overlap':float}, usecols=[0,1,2])
    gt_df['direc'] = '+'
    gt_df['i1'] = [i1-1 for i1 in gt_df['i1']]
    gt_df['i2'] = [i2-1 for i2 in gt_df['i2']]
    thetas = get_thetas(gt_df, seq_lens)
    gt_df = gt_df[thetas > min_theta]
    return gt_df

def get_thetas(gt_df, seq_lens):
    thetas = []
    ovlps = gt_df['overlap'].values
    i1s = gt_df['i1'].to_numpy()
    i2s = gt_df['i2'].to_numpy()
    tot_lens = seq_lens[i1s] + seq_lens[i2s]
    thetas = ovlps / (tot_lens - ovlps)
    return np.array(thetas)


def get_pred_df(pred_path, min_val=0, one_idx=False, n_cols=4):
    pred_df = pd.read_csv(pred_path, sep='\t', header=None, names=['i1','i2','overlap','direc'], dtype={'i1':str,'i2':str,'overlap':float,'direc':str})
    pred_df = pred_df[pred_df['overlap'] > min_val]
    pred_df['i1'] = [int(i1)-one_idx for i1 in pred_df['i1']]
    pred_df['i2'] = [int(i2)-one_idx for i2 in pred_df['i2']]
    return pred_df


def get_overlaps(gt_df, pred_dfs, n_seq, rc=True): 
    '''returns [gt, pred1, pred2, ...]'''
    overlaps = {}
    n_col = len(pred_dfs) + 1
    gt_np = gt_df.to_numpy()
    for run_idx, pred_df in enumerate(pred_dfs):
        pred_np = pred_df.to_numpy(); 
        for line in pred_np:
            i1,i2 = sorted([line[0],line[1]])
            score,is_rc = line[2],line[3]
            key = (i1,i2,is_rc)
            if run_idx == 0 or key not in overlaps:
                cur_scores = np.zeros(n_col)
                cur_scores[run_idx+1] = score # idx+1 because gt is first
                overlaps[key] = cur_scores
            else:
                overlaps[key][run_idx+1] = score

    for line in gt_np:
        i1,i2 = sorted([line[0],line[1]])
        score,is_rc = line[2],line[3]
        key = (i1,i2,is_rc)
        if key not in overlaps:
            cur_scores = np.zeros(n_col)
            cur_scores[0] = score # gt is first
            overlaps[key] = cur_scores
        else:
            overlaps[key][0] = score
    
    overlaps = np.array(list(overlaps.values()))
    n_missing_pairs = int(n_seq*(n_seq-1)/(1 if rc else 2) - len(overlaps))
    overlaps = np.concatenate((overlaps, np.zeros((n_missing_pairs,n_col))), axis=0)
    return overlaps

def roc(pred_dfs, gt_df, names, dset, n_seq, rc=True):
    aucs,tprs,fprs = [],[],[]
    plt.figure(figsize=(5,4))
    overlaps = get_overlaps(gt_df, pred_dfs, n_seq, rc=rc)
    gt = overlaps[:,0] > 0
    
    for run_idx,name in tqdm(enumerate(names),leave=True): 
        pred = overlaps[:,run_idx+1] # +1 because gt is first
        fpr,tpr,thresh = roc_curve(gt, pred)
        auc = round(roc_auc_score(gt, pred),4)
        aucs.append(auc)
        tprs.append(tpr)
        fprs.append(fpr)

        plt.plot(
            fpr,
            tpr,
            '-',
            lw=2,
            label="%s, AUROC = %0.2f" % (name, auc),
        )
    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {dset}")
    plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')
    plt.show()
    return aucs,fprs,tprs

    
def prc(pred_dfs, gt_df, names, dset, n_seq, rc=True):   
    aucs,precs,recs = [],[],[]
    plt.figure(figsize=(5,4))
    overlaps = get_overlaps(gt_df, pred_dfs, n_seq, rc=rc)
    gt = overlaps[:,0] > 0
    for run_idx,name in tqdm(enumerate(names),leave=True): 
        pred = overlaps[:,run_idx+1] # +1 because gt is first
        precision, recall, thresholds = precision_recall_curve(gt, pred)
        auc = round(average_precision_score(gt, pred),4)
        aucs.append(auc)
        precs.append(precision)
        recs.append(recall)
        if run_idx==0:
            plt.plot(
                recall,
                precision,
                '-',
                lw=2,
                label="%s, AUPRC = %0.2f" % (name, auc),
            )
        else:
            plt.plot(
                recall,
                precision,
                'o',
                markersize=2, 
                lw=2,
                label="%s, AUPRC = %0.2f" % (name, auc),
            )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PRC for {dset}")
    plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')         
    plt.show()
    return aucs,precs,recs


def pr_vs_k(pred_dfs, gt_df, ks, title, thresh=300):
    precs,recalls = [],[]
    plt.figure()
    for pred_df in pred_dfs:
        overlaps = get_overlaps(pred_df, gt_df)
        gt = overlaps[:,1] > 0 
        pred = overlaps[:,0] > thresh
        precision = precision_score(gt,pred)
        recall = recall_score(gt, pred)
        precs.append(precision)
        recalls.append(recall)

    plt.plot(ks, precs, '--', lw=2, label="precision")
    plt.plot(ks, recalls,'--', lw=2, label="recall")
    
#     plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("k")
    plt.ylabel("Precision/Recall")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    return precs,recalls

def roc_and_prc(pred_dfs, gt_df, names, dset, n_seq, rc=True):
    aurocs,fprs,tprs,auprcs,precs,recalls = [],[],[],[],[],[]
    plt.figure(figsize=(10,4))
    overlaps = get_overlaps(gt_df, pred_dfs, n_seq, rc=rc)
    gt = overlaps[:,0] > 0
    
    plt.subplot(121)
    for run_idx,name in tqdm(enumerate(names),leave=True): 
        pred = overlaps[:,run_idx+1] # +1 because gt is first
        fpr,tpr,thresh = roc_curve(gt, pred)
        auroc = round(roc_auc_score(gt, pred),4)
        aurocs.append(auroc)
        fprs.append(fpr)
        tprs.append(tpr)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label="%s, AUROC = %0.2f" % (name, auroc),
        )
    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {dset}")
    plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')         

    plt.subplot(122)
    for run_idx,name in tqdm(enumerate(names),leave=True): 
        pred = overlaps[:,run_idx+1] # +1 because gt is first
        precision, recall, thresholds = precision_recall_curve(gt, pred)
        auprc = round(average_precision_score(gt, pred),4)
        auprcs.append(auprc)
        precs.append(precision)
        recalls.append(recall)

        plt.plot(
            recall,
            precision,
            'o',
            markersize=3,
            lw=2,
            label="%s, AUPRC = %0.2f" % (name, auprc),
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PRC for {dset}")
    plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')         
    plt.show()
    return aurocs,fprs,tprs, auprcs,precs,recalls


def pr_vs_k(pred_dfs, gt_df, ks, title, thresh=300):
    precs,recalls = [],[]
    plt.figure()
    for pred_df in pred_dfs:
        overlaps = get_overlaps(pred_df, gt_df)
        gt = overlaps[:,1] > 0 
        pred = overlaps[:,0] > thresh
        precision = precision_score(gt,pred)
        recall = recall_score(gt, pred)
        precs.append(precision)
        recalls.append(recall)

    plt.plot(ks, precs, '--', lw=2, label="precision")
    plt.plot(ks, recalls,'--', lw=2, label="recall")
    
#     plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("k")
    plt.ylabel("Precision/Recall")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    return precs,recalls


# def roc_and_prc(pred_dfs, gt_df, names, dset):
#     aurocs,fprs,tprs = [],[],[]
#     plt.figure(figsize=(10,4))
    
#     plt.subplot(121)
#     for pred_df,name in tqdm(zip(pred_dfs,names),leave=True):
#         overlaps = get_overlaps(pred_df, gt_df)
#         gt = overlaps[:,1] > 0 
#         pred = overlaps[:,0]
#         fpr,tpr,thresh = roc_curve(gt, pred)
#         auroc = round(roc_auc_score(gt, pred),4)
#         aurocs.append(auroc)
#         fprs.append(fpr)
#         tprs.append(tpr)

#         plt.plot(
#             fpr,
#             tpr,
#             lw=2,
#             label="%s, AUROC = %0.2f" % (name, auroc),
#         )
    
#     plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"ROC for {dset}")
#     plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')         

#     plt.subplot(122)
#     auprcs,precs,recalls = [],[],[]
#     for pred_df,name in zip(pred_dfs,names):
#         overlaps = get_overlaps(pred_df, gt_df)
#         gt = overlaps[:,1] > 0 
#         pred = overlaps[:,0]
#         precision, recall, thresholds = precision_recall_curve(gt, pred)
#         precision, recall = precision[1:], recall[1:]
#         auprc = round(average_precision_score(gt, pred),4)
#         auprcs.append(auprc)
#         precs.append(precision)
#         recalls.append(recall)

#         plt.plot(
#             recall,
#             precision,
#             lw=2,
#             label="%s, AUPRC = %0.2f" % (name, auprc),
#         )
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(f"PRC for {dset}")
#     plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left')         
#     plt.show()
#     return aurocs,fprs,tprs, auprcs,precs,recalls



# def get_overlaps(pred_df, gt_df, n_seq): 
#     '''returns [pred, gt]'''
#     overlaps = np.full((n_seq,n_seq,2), 0)
#     for i in range(len(pred_df)):
#         line = pred_df.iloc[i]
#         if line.i2<line.i1: continue
#         overlaps[line.i1,line.i2,0] = line.overlap

#     for i in range(len(gt_df)):
#         line = gt_df.iloc[i]
#         if line.i2<line.i1: continue
#         overlaps[line.i1,line.i2,1] = line.overlap
    
#     overlaps = overlaps.reshape(n_seq**2, 2)
#     overlaps = overlaps[np.where(np.logical_or(overlaps[:,0]!=0, overlaps[:,1]!=0))]
#     return overlaps

# def get_overlaps(pred_df, gt_df): 
#     '''returns [pred, gt]'''
#     overlaps = {}
#     pred_np = pred_df.to_numpy(); gt_np = gt_df.to_numpy()
#     for line in pred_np:
#         i1,i2 = sorted([line[0],line[1]])
#         overlaps[(i1,i2,line[3])] = [line[2],0]

#     for line in gt_np:
#         i1,i2 = sorted([line[0],line[1]])
#         key = (i1,i2,line[3])
#         if key in overlaps:  
#             overlaps[key] = [overlaps[key][0],line[2]]
#         else:
#             overlaps[key] = [0,line[2]]
    
#     overlaps = np.array(list(overlaps.values()))
#     return overlaps



# def roc(pred_dfs, gt_df, names, title):
#     aurocs,fprs,tprs = [],[],[]
#     plt.figure()
#     for pred_df,name in tqdm(zip(pred_dfs,names),leave=True):
#         overlaps = get_overlaps(pred_df, gt_df)
#         gt = overlaps[:,1] > 0 
#         pred = overlaps[:,0]
#         fpr,tpr,thresh = roc_curve(gt, pred)
#         auroc = round(roc_auc_score(gt, pred),4)
#         aurocs.append(auroc)
#         fprs.append(fpr)
#         tprs.append(tpr)

#         plt.plot(
#             fpr,
#             tpr,
#             lw=2,
#             label="%s, AUROC = %0.2f" % (name, auroc),
#         )
    
#     plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.show()
#     return aurocs,fprs,tprs
    
# def pr_curves(pred_dfs, gt_df, names, title):
#     auprcs,precs,recalls = [],[],[]
#     plt.figure()
#     for pred_df,name in zip(pred_dfs,names):
#         overlaps = get_overlaps(pred_df, gt_df)
#         gt = overlaps[:,1] > 0 
#         pred = overlaps[:,0]
#         precision, recall, thresholds = precision_recall_curve(gt, pred)
#         auprc = round(average_precision_score(gt, pred),4)
#         auprcs.append(auprc)
#         precs.append(precision)
#         recalls.append(recall)

#         plt.plot(
#             recall,
#             precision,
#             lw=2,
#             label="%s, AUPRC = %0.2f" % (name, auprc),
#         )
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     plt.show()
#     return auprcs,precs,recalls