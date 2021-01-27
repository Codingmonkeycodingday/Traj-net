import os
import zipfile
import numpy as np
import torch
import pygtrie


def load_data(pathNum,pathLen):

    features=np.load("data/chengdu/features.npy")

    roadids = []

    districtids = []
    with open("data/chengdu/city_district.txt") as fr:
        for l in fr:
            temp = l.split("\t")
            id = temp[0]
            districtids.append(id)

    dic = {}
    with open("data/chengdu/boundary_en_chengdu.txt") as fr:
        for l in fr:
            temp = l.split("\t")
            id = temp[0]
            if id not in districtids:
                roadids.append(id)
    features = features[:, np.where(np.count_nonzero(features, axis=0) >= 3000)[0]]
    validRoadIds = np.array(roadids)[np.where(np.count_nonzero(features, axis=0) >= 3000)[0]].tolist()

    col_mean = np.true_divide(features.sum(0), (features != 0).sum(0))
    col_max=features.max(0)
    zeroInds = np.where(~np.isnan(col_mean))[0]
    inds = np.where(features == 0)
    features[inds] = np.take(col_max, inds[1])
    features = features[:, zeroInds]

    trajSamples=[]
    with open("data/chengdu/s2osm-traj-cd.txt") as fr:
        for l in fr:
            temp = l.rstrip('\n').split(",")
            indices = [(i, i + pathLen) for i in range(len(temp) - (pathLen+ 1))]
            for i, j in indices:
                trajSamples.append(temp[i: i + pathLen])
    trajSamples=np.array(trajSamples)
    lnodes=trajSamples[:,-1]
    lnodes=np.unique(np.array(lnodes))

    r2sDic={}
    s2rDic={}
    fixSet=set(lnodes)


    X=[]
    ind=0
    print(features.shape)
    with open("data/chengdu/s2osm-st-cd.txt") as fr:
        for l in fr:
            temp = l.rstrip('\n').split("\t")
            instersect=set(temp[1].split(",")).intersection(fixSet)
            instersect=list(instersect)
            if temp[0] in validRoadIds and len(instersect)>0:
                for t in instersect:
                  if t not in s2rDic:
                    s2rDic[t]=ind
                r2sDic[ind]=instersect
                X.append(features[:,validRoadIds.index(temp[0])])
                ind+=1

    keys=[*s2rDic]
    newS2Rdic={}
    newR2SDic={}

    for i in range(len(keys)):
      newS2Rdic[i]=s2rDic[keys[i]]
    s2rDic=newS2Rdic
    curIndex=len(s2rDic)
    for road in r2sDic:
      intersections=r2sDic[road]
      temp=[keys.index(ist) for ist in intersections if s2rDic[keys.index(ist)]==road]
      if len(temp)>0:
        newR2SDic[road]=temp
      else:
        newR2SDic[road]=[curIndex]
        s2rDic[curIndex]=road
        curIndex+=1
    r2sDic=newR2SDic

    allTrajs=np.load("data/chengdu/traj-cd-57.npy",allow_pickle=True)
    trajDic={}
    for traj in allTrajs:
      if traj[-1] not  in trajDic:
        trajDic[traj[-1]]=[traj]
      else:
        trajDic[traj[-1]].append(traj)
    for traj1 in trajDic:
      trajDic[traj1]=np.array(trajDic[traj1]).astype(int)


    X=np.array(X)
    X=X.transpose((1, 0))
    X=np.expand_dims(X, axis=2)
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    return X,r2sDic,s2rDic,trajDic,keys



def random_trajs(trajDic,s2rDic,keys,pathNum):
    ind=0
    t = pygtrie.StringTrie()
    roots=[]
    # print(keys)
    randomtrajs={}
    for key in keys:
        #filter = np.asarray([key])
        trajSamples=trajDic[key]
        subTraj=trajSamples[np.random.choice(trajSamples.shape[0], pathNum)]
        subTraj=subTraj.tolist()
        for temp1 in subTraj:
          tempTraj=[]
          for seg in temp1:
            if str(seg) in keys:
              tempTraj.append(str(keys.index(str(seg))))
          if tempTraj[0] not in roots:
            roots.append(tempTraj[0])
          #print(tempTraj)
          if tempTraj[-1] not in randomtrajs:
            randomtrajs[tempTraj[-1]]=[tempTraj]
          else:
            randomtrajs[tempTraj[-1]].append(tempTraj)
          for i in range(len(tempTraj)-1):
              if t.has_key("/".join(tempTraj[:i+1])):
                if tempTraj[i+1] not in t["/".join(tempTraj[:i+1])]:
                  t["/".join(tempTraj[:i+1])].append(tempTraj[i+1])
              else:
                t["/".join(tempTraj[:i+1])]=[tempTraj[i+1]]
        ind+=1

    mask=np.zeros((len(s2rDic), len(s2rDic)))
    for n in randomtrajs:
      for j in range(len(randomtrajs[n])):
        temp=randomtrajs[n][j]
        for i in range(len(temp)-1):
          #if temp[0][1:-1] in keys and temp[1][1:-1] in keys:
          mask[int(temp[i])][int(temp[i+1])]=1
    mask = mask.astype(np.float32)
    return t,roots,randomtrajs,mask


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    td=24*6
    tw=td*7
    cInterval=3
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(tw,X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target,daily,weekly,cFeatures = [], [],[],[],[]


    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        cFeatures.append(
            X[:, :, i + num_timesteps_input-cInterval*6:i + num_timesteps_input-cInterval:cInterval ].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])
        dailyH=[X[:, :, i + num_timesteps_input-(k+1)*td: j-(k+1)*td] for k in range(2)]

        daily.append(np.concatenate(dailyH,axis=2).transpose(
                (0, 2, 1)))

        weeklyH=X[:, :, i + num_timesteps_input-tw: j-tw]

        weekly.append(weeklyH.transpose(
                (0, 2, 1)))
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(daily)), \
           torch.from_numpy(np.array(weekly)), \
           torch.from_numpy(np.array(cFeatures)),\
           torch.from_numpy(np.array(target))

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

