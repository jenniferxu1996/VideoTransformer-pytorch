import numpy as np
import torch
from torch import nn
import data_transform as T
from dataset import DecordInit
from video_transformer import ViViT
from einops import rearrange


class ViViTMLP(nn.Module):

    def __init__(self, vivit_ckpt, img_size, num_frames, mlp_nodes):
        super().__init__()
        self.vivit = ViViT(pretrain_pth=vivit_ckpt,
                           img_size=img_size,
                           num_frames=num_frames)
        self.mlp = nn.Linear(self.vivit.embed_dims, mlp_nodes)
        mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
        self.data_transform = T.Compose([
            T.Resize(scale_range=(-1, 256)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def process_video(self, videos):
        """
        videos: shape [BATCH, FRAMES, CHANNEL, HEIGHT, WIDTH]
        """
        self.data_transform.randomize_parameters()
        batch_size = videos.shape[0]
        videos = rearrange(videos, 'b t c h w -> (b t) c h w')
        videos = self.data_transform(videos)
        return rearrange(videos, '(b t) c h w -> b t c h w', b=batch_size)

    def forward(self, videos):
        with torch.no_grad():
            x = self.vivit(videos)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    device = torch.device("cpu")

    # params
    video_path = './demo/YABnJL_bDzw.mp4'
    num_frames = 16
    frame_interval = 16
    img_size = (224, 224)
    mlp_nodes = 512
    ckpt_path = "pretrained/vit_base_patch16_224.pth"

    video_decoder = DecordInit()
    temporal_sample = T.TemporalRandomCrop(num_frames * frame_interval)

    model = ViViTMLP(ckpt_path, img_size, num_frames, mlp_nodes).to(device)

    # read video data
    v_reader = video_decoder(video_path)
    total_frames = len(v_reader)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    if end_frame_ind - start_frame_ind < num_frames:
        raise ValueError(f'the total frames of the video {video_path} is less than {num_frames}')

    frame_indice = np.linspace(0, end_frame_ind - start_frame_ind - 1, num_frames, dtype=int)
    video = v_reader.get_batch(frame_indice).asnumpy()
    del v_reader
    video = torch.from_numpy(video).permute(0, 3, 1, 2)  # Video transform: T C H W
    videos = video.unsqueeze(0).to(device)  # shape: [1, T, C, H, W]
    labels = torch.rand((1, mlp_nodes)).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # train
    for epoch in range(5):
        # preprocess videos first
        x = model.process_video(videos)
        # feed x into the model
        y = model(x)

        loss = loss_fn(y, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.detach().cpu().numpy())
