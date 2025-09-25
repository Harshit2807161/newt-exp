import os

import torch
from torchvision.io import read_image

from common import TASK_SET
from common.vision_encoder import PretrainedEncoder

RECOMPUTE = False  # Set to True to recompute features

# Load encoder
encoder = PretrainedEncoder()

# for task in TASK_SET["metaworld"]:
for task in TASK_SET['soup']:

    # Check whether features have already been computed
    td_path = f"/data/nihansen/code/tdmpc25/data/{task}.pt"
    if os.path.exists(td_path):
        td = torch.load(td_path, weights_only=False)
        if 'feat' in td and 'feat-stacked' in td and not RECOMPUTE:
            print(f"Features already computed for task {task}. Skipping.")
            continue

    # Load image data
    print('Encoding data for task:', task)

    i = 0
    fp = lambda i: f"/data/nihansen/code/tdmpc25/data/{task}-{i}.png"
    features = []

    while os.path.exists(fp(i)):
        frames = read_image(fp(i))  # (3, 224, 224*B)
        num_frames = frames.shape[-1] // 224  # Number of images in batch
        frames = frames.view(3, 224, num_frames, 224)  # Reshape to (3, 224, B, 224)
        frames = frames.permute(2, 0, 1, 3)  # Reshape to (B, 3, 224, 224)
        
        # Encode frames in smaller batches
        batch_size = 256
        frame_idx = 0
        while frame_idx < num_frames:
            # Extract batch of frames
            end_idx = frame_idx + batch_size
            if end_idx > num_frames:
                end_idx = num_frames
            batch_frames = frames[frame_idx:end_idx]
            out = encoder(batch_frames)
            features.append(out.cpu())
            print(f'Processed {end_idx}/{num_frames} frames in chunk {i+1} for task {task}, feature dim: {int(out.shape[-1])}')
            frame_idx += batch_size

        i += 1

    if len(features) == 0:
        print(f"No data found for task {task}. Skipping.")
        continue

    # Save features
    features = torch.cat(features, dim=0)  # Concatenate all features
    print(features.shape, features.dtype, features.device)
    features = features[:td['obs'].shape[0]]  # Match number of observations
    print('Final feature shape:', features.shape)
    td['feat'] = features
    # torch.save(td, f"/data/nihansen/code/tdmpc25/data/{task}.pt")

    def frame_stack_feat(td, k=3):
        feats = td['feat']  # [N, D]
        episodes = td['episode']  # [N]
        N, D = feats.shape
        device = feats.device

        # Prepare output tensor
        stacked = torch.empty(N, k, D, device=device)

        # Find episode boundaries
        episode_ids = episodes.unique_consecutive()
        for eid in episode_ids:
            idx = (episodes == eid).nonzero(as_tuple=False).squeeze(-1)
            feat_ep = feats[idx]  # [T, D]
            T = feat_ep.shape[0]

            # Pad front with first frame
            pad = feat_ep[0].unsqueeze(0).expand(k - 1, -1)  # [k-1, D]
            padded = torch.cat([pad, feat_ep], dim=0)  # [T + k - 1, D]

            # Collect sliding windows
            stacked_ep = torch.stack([padded[i: i + T] for i in range(k)], dim=1)  # [T, k, D]

            # Assign back
            stacked[idx] = stacked_ep

        # Flatten last two dims
        td['feat-stacked'] = stacked.reshape(N, k * D)
        return td
    
    td = frame_stack_feat(td, k=3)
    torch.save(td, f"/data/nihansen/code/tdmpc25/data/{task}.pt")
