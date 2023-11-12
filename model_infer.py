import os

import numpy as np
import pandas as pd
import torch
import tqdm

import data_transform as T
from dataset import DecordInit
from video_transformer import ViViT
from transformer import ClassificationHead
from sklearn.metrics import classification_report

VALID = ['s21-d21-cam-002', 's21-d23-cam-002', 's21-d27-cam-002', 's21-d28-cam-002', 's21-d29-cam-002', 's21-d35-cam-002',
         's21-d39-cam-002', 's21-d40-cam-002', 's21-d42-cam-002', 's21-d43-cam-002', 's21-d45-cam-002', 's21-d49-cam-002',
         's21-d50-cam-002', 's21-d52-cam-002', 's21-d53-cam-002', 's21-d55-cam-002', 's21-d63-cam-002']
TEST = ['s22-d23-cam-002', 's22-d25-cam-002', 's22-d26-cam-002', 's22-d29-cam-002', 's22-d31-cam-002',
        's22-d34-cam-002', 's22-d35-cam-002', 's22-d43-cam-002', 's22-d46-cam-002', 's22-d48-cam-002', 's22-d53-cam-002',
        's22-d55-cam-002', 's28-d23-cam-002', 's28-d25-cam-002', 's28-d27-cam-002', 's28-d39-cam-002', 's28-d46-cam-002',
        's28-d51-cam-002', 's28-d70-cam-002', 's28-d74-cam-002', 's29-d29-cam-002', 's29-d31-cam-002', 's29-d39-cam-002',
        's29-d42-cam-002', 's29-d49-cam-002', 's29-d50-cam-002', 's29-d52-cam-002', 's29-d71-cam-002', 's33-d23-cam-002',
        's33-d27-cam-002', 's33-d45-cam-002', 's33-d49-cam-002', 's33-d50-cam-002', 's33-d54-cam-002', 's34-d21-cam-002',
        's34-d28-cam-002', 's34-d34-cam-002', 's34-d40-cam-002', 's34-d41-cam-002', 's34-d63-cam-002', 's34-d69-cam-002',
        's34-d73-cam-002']

def infer_sample(video_path, video_annotations, model, cls_head, temporal_sample, video_decoder, device, cls_map, num_frames=16, batch_size=8):
    # load video data and apply data preprocessing methods
    v_reader = video_decoder(video_path)

    videos = []
    for index, row in video_annotations.iterrows():
        start_frame_ind, end_frame_ind = temporal_sample(int(row['startFrame']), int(row['endFrame']))
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
        video = v_reader.get_batch(frame_indice).asnumpy()
        video = torch.from_numpy(video).permute(0,3,1,2) # Video transform: T C H W
        data_transform.randomize_parameters()
        video = data_transform(video)
        videos.append(video)
    del v_reader

    targets = []
    preds = []
    for i in range(0, len(videos), batch_size):
        last = min(i+batch_size, len(videos))
        batch = torch.stack(videos[i: last]).to(device)
        # compute feature from last layer
        pred = cls_head(model(batch)).detach().cpu().numpy()
        for index in range(i, last):
            label = video_annotations.iloc[index]['activity']
            target = cls_map[label]
            p = np.argmax(pred[index-i])
            targets.append(target)
            preds.append(p)
    return targets, preds


def replace_state_dict(state_dict):
    for old_key in list(state_dict.keys()):
        if old_key.startswith('model'):
            new_key = old_key[6:] # skip 'model.'
            if 'in_proj' in new_key:
                new_key = new_key.replace('in_proj_', 'qkv.') #in_proj_weight -> qkv.weight
            elif 'out_proj' in new_key:
                new_key = new_key.replace('out_proj', 'proj')
            state_dict[new_key] = state_dict.pop(old_key)
        else: # cls_head
            new_key = old_key[9:]
            state_dict[new_key] = state_dict.pop(old_key)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = "pretrained/ep_112_top1_acc_0.510.pth"
    weights_from = "kinetics"  # imagenet
    mode = "test"
    num_frames = 16
    img_size = (224, 224)
    frame_interval = 16
    batch_size = 4
    num_class = 87

    raw_video_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_structure_for_har', f'data/mpii_cooking/raw')
    annotation_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_structure_for_har', f'data/mpii_cooking/annotation.csv')
    class_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_structure_for_har', f'data/mpii_cooking/classes.txt')

    cls_map = {}
    with open(class_path, 'r') as f:
        for i, cls in enumerate(f):
            cls_map[cls.strip()] = i
    print(cls_map)

    # load the trained model
    model = ViViT(pretrain_pth=ckpt_path,
                  img_size=img_size,
                  num_frames=num_frames,
                  weights_from=weights_from)
    model.eval()
    model = model.to(device)
    cls_head = ClassificationHead(num_class, model.embed_dims)
    state_dict = torch.load(ckpt_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    replace_state_dict(state_dict)
    msg = cls_head.load_state_dict(state_dict, strict=False)
    print(msg)
    cls_head.eval()
    cls_head = cls_head.to(device)

    # declare data preprocessing methods
    mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
    data_transform = T.Compose([
            T.Resize(scale_range=(-1, 256)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std)
            ])
    temporal_sample = T.TemporalRandomCropWithRange(num_frames * frame_interval)
    video_decoder = DecordInit()

    video_clips = os.listdir(raw_video_path)
    if mode == 'test':
        video_clips = [each_sample for each_sample in os.listdir(raw_video_path) if os.path.splitext(each_sample)[0] in TEST]
    elif mode == 'valid':
        video_clips = [each_sample for each_sample in os.listdir(raw_video_path) if os.path.splitext(each_sample)[0] in VALID]
    else:
        video_clips = [each_sample for each_sample in os.listdir(raw_video_path) if os.path.splitext(each_sample)[0] not in VALID + TEST]
    annotations = pd.read_csv(annotation_path)

    all_preds = []
    all_targets = []
    for video_clip in tqdm.tqdm(video_clips, desc='process video'):
        video_name = os.path.splitext(video_clip)[0]
        video_annotation = annotations.loc[annotations['fileName'] == video_name]
        video_clip_path = os.path.join(raw_video_path, video_clip)

        targets, preds = infer_sample(video_clip_path, video_annotation, model, cls_head, temporal_sample, video_decoder, device, cls_map, num_frames=num_frames, batch_size=batch_size)
        all_preds.append(preds)
        all_targets.append(targets)

    print(classification_report(np.concatenate(all_targets), np.concatenate(all_preds)))


