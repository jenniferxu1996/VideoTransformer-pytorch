import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm

import data_transform as T
from dataset import DecordInit
from video_transformer import ViViT
from transformer import ClassificationHead
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, multilabel_confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def get_average_precision_value(y_true, y_prob, class_num):
    precision = {}
    recall = {}
    ap = {}
    print(y_true.shape, y_prob.shape)
    for i in range(class_num):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        # print(type(precision[i]), type(recall[i]), type(ap[i]))
        precision[i] = list(precision[i])
        recall[i] = list(recall[i])
        ap[i] = float(ap[i])
    return precision, recall, ap


def get_evaluation_metrics(y_true, y_prob, y_predict, class_num):
    # Get all the evaluation metrics for the inference data sample
    precision_all, recall_all, ap = get_average_precision_value(y_true, y_prob, class_num)
    # Convert the prediction sequence to one-hot representation
    y_predict_onehot = label_binarize(y_predict, classes=[i for i in range(class_num)])
    multi_confusion = multilabel_confusion_matrix(y_true, y_predict_onehot)
    # print(f"the shape of multi_confusion = {len(multi_confusion)}")
    # Convert the one-hot representation for ground truth to the label sequence
    y_true_sequence = []
    for i in range(len(y_true)):
        for label in range(0, class_num):
            if y_true[i][label] == 1:
                y_true_sequence.append(label)
    # print(y_true_sequence)
    labels = [i for i in range(0, class_num)]
    result_report = classification_report(y_true_sequence, y_predict, labels=labels)
    confusion = confusion_matrix(y_true_sequence, y_predict, labels=labels)
    mcc = matthews_corrcoef(y_true_sequence, y_predict)
    f1 = f1_score(y_true_sequence, y_predict, average="weighted")
    acc = accuracy_score(y_true_sequence, y_predict)
    # print(type(precision), type(recall), type(ap), type(multi_confusion), type(confusion))
    return precision_all, recall_all, ap, result_report, multi_confusion.tolist(), confusion.tolist(), mcc, acc, f1

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
    probs = []
    for i in range(0, len(videos), batch_size):
        last = min(i+batch_size, len(videos))
        batch = torch.stack(videos[i: last]).to(device)
        # compute feature from last layer
        pred = cls_head(model(batch)).detach().cpu().numpy()
        for index in range(i, last):
            label = video_annotations.iloc[index]['activity']
            target = cls_map[label]
            p = np.argmax(pred[index-i])
            targets.append([1 if target == j else 0 for j in range(num_class)])
            preds.append(p)
            probs.append(pred[index-i])
    return targets, preds, probs


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
    all_probs = []
    for video_clip in tqdm.tqdm(video_clips, desc='process video'):
        video_name = os.path.splitext(video_clip)[0]
        video_annotation = annotations.loc[annotations['fileName'] == video_name]
        video_clip_path = os.path.join(raw_video_path, video_clip)

        targets, preds, probs = infer_sample(video_clip_path, video_annotation, model, cls_head, temporal_sample, video_decoder, device, cls_map, num_frames=num_frames, batch_size=batch_size)
        all_preds.append(preds)
        all_targets.append(targets)
        all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    precision, recall, ap, result_report, multi_confusion, confusion, mcc, acc, f1 = get_evaluation_metrics(all_targets, all_probs, all_preds, num_class)

    with open(f'result.txt', 'w') as f:
        print("average precision:", ap, file=f)
        print("result report:", result_report, file=f)
        print("confusion matrix for multiclassification:", multi_confusion, file=f)
        print("confusion matrix:", confusion, file=f)
        print("matthews coefficient:", mcc, file=f)
        print("acc:", acc, file=f)
        print("macro f1:", f1, file=f)
