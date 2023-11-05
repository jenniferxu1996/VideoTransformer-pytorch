import os

import numpy as np
import pandas as pd
import torch

import data_transform as T
from dataset import DecordInit
from video_transformer import ViViT
import weight_init


def load_last_feature_for_a_sample(video_path, model, temporal_sample, video_decoder, device, num_frames=16):
    # load video data and apply data preprocessing methods
    v_reader = video_decoder(video_path)
    total_frames = len(v_reader)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    if end_frame_ind-start_frame_ind < num_frames:
        raise ValueError(f'the total frames of the video {video_path} is less than {num_frames}')
    frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, num_frames, dtype=int)
    video = v_reader.get_batch(frame_indice).asnumpy()
    del v_reader

    video = torch.from_numpy(video).permute(0,3,1,2) # Video transform: T C H W
    data_transform.randomize_parameters()
    video = data_transform(video)
    video = video.unsqueeze(0).to(device)

    # compute feature from last layer
    logits = model(video)
    print(logits.shape)
    return logits


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = "pretrained/vit_base_patch16_224.pth"
    # load the trained model
    num_frames = 16
    img_size = (224, 224)
    frame_interval = 16
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
    temporal_sample = T.TemporalRandomCrop(num_frames*frame_interval)
    video_decoder = DecordInit()

    video_clips_path = f"video_clip"
    video_clips = os.listdir(video_clips_path)

    vivit_feature_df = pd.DataFrame(columns=['sample_id', 'features'])
    for video_clip in video_clips:
        video_clip_path = os.path.join(video_clips_path, video_clip)
        logits = load_last_feature_for_a_sample(video_clip_path, model, temporal_sample, video_decoder, device)
        feature_dic = {'sample_id': os.path.split(video_clip)[0], 'features': logits}
        vivit_feature_df = pd.concat([ vivit_feature_df, pd.DataFrame([feature_dic])], ignore_index=True)
    print(vivit_feature_df)
    vivit_feature_df.to_pickle('vivit_feature.pkl')



