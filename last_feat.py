import os

import numpy as np
import pandas as pd
import torch
import tqdm

import data_transform as T
from dataset import DecordInit
from video_transformer import ViViT


def load_last_feature_for_a_sample(video_name, video_path, video_annotations, model, temporal_sample, video_decoder, device, num_frames=16, batch_size=8):
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

    feature_dics = []
    for i in range(0, len(videos), batch_size):
        last = min(i+batch_size, len(videos))
        batch = torch.stack(videos[i: last]).to(device)
        # compute feature from last layer
        logits = model(batch).detach().cpu().numpy()
        for index in range(i, last):
            feature_dic = {'sample_id': f'{video_name}_{video_annotations.iloc[index]["startFrame"]}_{video_annotations.iloc[index]["endFrame"]}', 'features': logits[index-i]}
            feature_dics.append(feature_dic)
    return feature_dics


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = "pretrained/vit_base_patch16_224.pth"
    num_frames = 16
    img_size = (224, 224)
    frame_interval = 16
    batch_size = 4

    raw_video_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_structure_for_har', f'data/mpii_cooking/raw')
    annotation_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_structure_for_har', f'data/mpii_cooking/annotation.csv')

    # load the trained model
    model = ViViT(pretrain_pth=ckpt_path,
                  img_size=img_size,
                  num_frames=num_frames)
    model.eval()
    model = model.to(device)
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
    annotations = pd.read_csv(annotation_path)

    vivit_feature_df = pd.DataFrame(columns=['sample_id', 'features'])
    for video_clip in tqdm.tqdm(video_clips, desc='process video'):
        video_name = os.path.splitext(video_clip)[0]
        video_annotation = annotations.loc[annotations['fileName'] == video_name]
        video_clip_path = os.path.join(raw_video_path, video_clip)

        feature_dics = load_last_feature_for_a_sample(video_name, video_clip_path, video_annotation, model, temporal_sample, video_decoder, device, batch_size=batch_size)
        vivit_feature_df = pd.concat([vivit_feature_df, pd.DataFrame(feature_dics)], ignore_index=True)

    vivit_feature_df.to_pickle('vivit_feature.pkl')



