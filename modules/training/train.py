"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--rf100_root_path', type=str, default='/path/to/rf100_dataset',
                        help='Path to the RF100 dataset root directory.')
    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth', 'xfeat_rf100'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to the checkpoint to resume from.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.rf100_underwater import RF100UnderwaterDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path,
                       rf100_root_path,
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_default',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500, resume_ckpt=None):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        # Load checkpoint if resume_ckpt is provided
        self.start_step = 0
        if resume_ckpt:
            checkpoint = torch.load(resume_ckpt, map_location=self.dev)
            self.net.load_state_dict(checkpoint['model_state_dict'])

            # Extract step from checkpoint filename
            match = re.search(r'_(\d+)\.pth$', resume_ckpt)
            if match:
                self.start_step = int(match.group(1))
                print(f"Resumed training from checkpoint {resume_ckpt} at step {self.start_step}")
            else:
                print(f"Warning: Could not extract step from checkpoint filename: {resume_ckpt}")

            # Attempt to load optimizer state (if it exists)
            if 'optimizer_state_dict' in checkpoint:
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer state loaded from {resume_ckpt}")
            else:
                print("Warning: No optimizer state found in checkpoint. Optimizer will start fresh.")

            # Attempt to load scheduler state (if it exists)
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Scheduler state loaded from {resume_ckpt}")

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 3,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data_md = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader_md = DataLoader(data_md,
                                          batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                          shuffle=True)
            self.data_iter_md = iter(self.data_loader_md)

        else:
            self.data_iter_md = None
        ##################### MEGADEPTH INIT END #######################


        ##################### RF100 INIT ##########################
        #if model_name in ('xfeat_default', 'xfeat_rf100'):
            #data_rf100 = RF100UnderwaterDataset(root_dir=rf100_root_path, split='train')

            #self.data_loader_rf100 = DataLoader(data_rf100,
                                                #batch_size=int(self.batch_size * 0.3 if model_name == 'xfeat_default' else batch_size),
                                                #shuffle=True)
            #self.data_iter_rf100 = iter(self.data_loader_rf100)
        #else:
            #self.data_iter_rf100 = None
        ##################### RF100 INIT END #######################


        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name


    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d_md, d_rf100 = None, None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        if self.data_iter_md is not None:
            d_md = next(self.data_iter_md)

        if self.data_iter_rf100 is not None:
            d_rf100 = next(self.data_iter_rf100)

        with tqdm.tqdm(total=self.steps) as pbar:
            pbar.update(self.start_step) # Initialize progress bar to the correct start step
            for i in range(self.start_step, self.steps):
                if not self.dry_run:
                    if self.data_iter_md is not None:
                        try:
                            # Get the next MD batch
                            d_md = next(self.data_iter_md)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter_md = iter(self.data_loader_md)
                            d_md = next(self.data_iter_md)

                    if self.augmentor is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d_md is not None:
                    for k in d_md.keys():
                        if isinstance(d_md[k], torch.Tensor):
                            d_md[k] = d_md[k].to(self.dev)
                
                    p1_md, p2_md = d_md['image0'], d_md['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d_md, 8)

                #if d_rf100 is not None:
                      #for k in d_rf100.keys():
                        #if isinstance(d_rf100[k], torch.Tensor):
                            #d_rf100[k] = d_rf100[k].to(self.dev)

                    #p1_rf100, p2_rf100 = d_rf100['image0'], d_rf100['image1']
                    #positives_rf100_coarse = d_rf100['coarse_correspondences']

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                #Join megadepth & synthetic data
                with torch.inference_mode():
                    #RGB -> GRAY
                    if d_md is not None:
                        p1_md = p1_md.mean(1, keepdim=True)
                        p2_md = p2_md.mean(1, keepdim=True)
                    if d_rf100 is not None:
                        p1_rf100 = p1_rf100.mean(1, keepdim=True)
                        p2_rf100 = p2_rf100.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    #Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    elif self.model_name == 'xfeat_megadepth':
                        p1, p2 = p1_md, p2_md
                        positives_c = positives_md_coarse
                    else:  # xfeat_rf100
                        p1, p2 = p1_rf100, p2_rf100
                        positives_c = positives_rf100_coarse

                #Check if batch is corrupted with too few corresp  ondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                #Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    #Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    #Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    #grab heatmaps at corresponding idxs
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    #Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()


                # Save checkpoint
                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save({
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                pbar.set_description( 'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                                                                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, loss_coarse, loss_coord, loss_l1, nb_coarse, loss_kp_pos, acc_pos) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)



if __name__ == '__main__':

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path,
        rf100_root_path=args.rf100_root_path,
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        resume_ckpt=args.resume_ckpt
    )

    #The most fun part
    trainer.train()