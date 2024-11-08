import sys
import numpy as np
import torch
from dataloader_iemocap import IEMOCAPDataset
from dataloader_cmumosi import CMUMOSIDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model_expert_softmoe import MoMKE
sys.path.append('./')
import config

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers):
    ## CMU datasets
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum+valNum))
        test_idxs = list(range(trainNum+valNum, trainNum+valNum+testNum))

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=False)
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=dataset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=False)
        train_loaders = [train_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim


    ## IEMOCAP dataset
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: ## five folder cross-validation, each fold contains (train, test)
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder): # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        ## gain train and test loaders
        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      collate_fn=dataset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=False)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     num_workers=num_workers,
                                     pin_memory=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = MoMKE(args,
                  adim, tdim, vdim, D_e,
                  n_classes=args.n_classes,
                  depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                  attn_drop_rate=args.attn_drop_rate,
                  no_cuda=args.no_cuda)
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    return model


## follow cpm-net's masking manner
# def generate_mask(seqlen, batch):
#     """Randomly generate incomplete data information, simulate partial view data with complete view data
#     """
#     audio_mask = np.array([1, 1, 1, 0, 1, 0, 0])
#     text_mask = np.array([1, 1, 0, 1, 0, 1, 0])
#     visual_mask = np.array([1, 0, 1, 1, 0, 0, 1])
#
#     audio_mask = audio_mask.repeat(seqlen * batch / 7)
#     text_mask = text_mask.repeat(seqlen * batch / 7)
#     visual_mask = visual_mask.repeat(seqlen * batch / 7)
#
#     matrix = [audio_mask, text_mask, visual_mask]
#     return matrix
def generate_mask(seqlen, batch, test_condition, first_stage):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """
    if first_stage:
        audio_mask = np.array([1])
        text_mask = np.array([1])
        visual_mask = np.array([1])
    else:
        audio_mask = np.array([1 if 'a' in test_condition else 0])
        text_mask = np.array([1 if 't' in test_condition else 0])
        visual_mask = np.array([1 if 'v' in test_condition else 0])
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask = text_mask.repeat(seqlen * batch)
    visual_mask = visual_mask.repeat(seqlen * batch)

    matrix = [audio_mask, text_mask, visual_mask]
    return matrix


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features