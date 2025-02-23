from argparse import ArgumentParser
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from PIL import Image
from event_data import EventData
from model import EvINRModel

def config_parser():
    parser = ArgumentParser(description="EvINR")
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name')
    parser.add_argument('--data_path', '-d', type=str, help='Path of events.npy to train')
    parser.add_argument('--output_dir', '-o', type=str, default='logs', help='Directory to save output')
    parser.add_argument('--t_start', type=float, default=0, help='Start time')
    parser.add_argument('--t_end', type=float, default=5, help='End time')
    parser.add_argument('--H', type=int, default=180, help='Height of frames')
    parser.add_argument('--W', type=int, default=240, help='Width of frames')
    parser.add_argument('--color_event', action='store_true', default=False, help='Whether to use color event')
    parser.add_argument('--event_thresh', type=float, default=1, help='Event activation threshold')
    parser.add_argument('--train_resolution', type=int, default=20, help='Number of training frames')
    parser.add_argument('--val_resolution', type=int, default=20, help='Number of validation frames')
    parser.add_argument('--no_c2f', action='store_true', default=False, help='Whether to use coarse-to-fine training')
    parser.add_argument('--iters', type=int, default=2000, help='Training iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--net_layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--net_width', type=int, default=512, help='Hidden dimension of the network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser

def main(args):
    split_number = 5
    writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name))
    events = []
    model = []
    optimizer = []
    for i in range(2*split_number-1):
      diff_time = args.t_end-args.t_start
      events.append(EventData(
          args.data_path, i*diff_time//(2*split_number)+args.t_start, i*diff_time//(2*split_number)+diff_time//split_number+args.t_start, args.H, args.W, args.color_event, args.event_thresh, args.device))
      model.append(EvINRModel(
          args.net_layers, args.net_width, H=args.H, W=args.W, recon_colors=args.color_event
          ).to(args.device))
      optimizer.append(torch.optim.AdamW(params=model[i].net.parameters(), lr=3e-4))
      events[i].stack_event_frames(args.train_resolution)
      for i_iter in range(1, args.iters + 1):
        optimizer[i].zero_grad()
        log_intensity_preds = model[i](events[i].timestamps)
        model[i].get_losses(log_intensity_preds, events[i].event_frames).backward()
        optimizer[i].step()






    with torch.no_grad():
        accumulation_number = 0
        for i in range(2*split_number-1):
          if i == 0:
            val_timestamps = torch.linspace(0, 0.75, args.val_resolution).to(args.device).reshape(-1, 1)
          elif i == 2*split_number-2:
            val_timestamps = torch.linspace(0.25, 1, args.val_resolution).to(args.device).reshape(-1, 1)
          else:
            val_timestamps = torch.linspace(0.25, 0.75, args.val_resolution).to(args.device).reshape(-1, 1)
          log_intensity_preds = model[i](val_timestamps)
          intensity_preds = model[i].tonemapping(log_intensity_preds).squeeze(-1)
          for j in range(0, intensity_preds.shape[0]):
            intensity1 = intensity_preds[j].cpu().detach().numpy()
            image_data = (intensity1*255).astype(np.uint8)
            # 将 NumPy 数组转换为 PIL 图像对象
            image = Image.fromarray(image_data)
            output_path = os.path.join('/content/EvINR/logs', 'output_image_{}.png'.format(j+accumulation_number))
            image.save(output_path)
          accumulation_number = accumulation_number + intensity_preds.shape[0]
          print(accumulation_number)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)

