import argparse
import os
import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataloaders.transform import source_collate_fn_tr_fda, collate_fn_ts
from dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from dataloaders.convert_csv_to_list import convert_labeled_list

from PIL import Image
from sklearn.cluster import KMeans
from networks.unet import UNet
from plot.evaluate_mu import eval_print_all_CU
from utils.util import sigmoid_entropy
from utils.lr import adjust_learning_rate
from utils.normalize import normalize_image
from utils.loss import DiceLoss, smooth_loss

if __name__ == '__main__':
    target_dataset = "MESSIDOR_Base1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str,
                        default=r'D:\SFDA\ProSFDA-master\prosfda\log\UNet_Source_Model\checkpoints\model_final.model')
    parser.add_argument('--batchsize', type=int, default=16)  # 16
    parser.add_argument('-g', '--gpu', type=int, default=3)
    parser.add_argument('--initial_lr', type=float, default=1e-3, required=False, help='initial learning rate.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False, help='num_epochs.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False, help='gamma in feature alignment loss.')
    parser.add_argument('--root', default=r'D:\RIGAPlus')
    parser.add_argument('--tr_csv', help='training csv file.', default=[r'D:\RIGAPlus\RIGA-pseudo-our\{}\Unlabeled\generate_pseudo.csv'.format(target_dataset)])
    parser.add_argument('--ts_csv', help='test csv file.', default=[r'D:\RIGAPlus\{}.csv'.format(target_dataset)])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    inference_tag = target_dataset
    visualization_folder = r'D:\SFDA\SFDA-our\logs\visualization\PCDCL\{}'.format(target_dataset)
    os.makedirs(visualization_folder, exist_ok=True)
    initial_lr = args.initial_lr
    num_epochs = args.num_epochs
    gamma = args.gamma
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    root_folder = args.root
    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, img_normalize=False)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list)
    train_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                               batch_size=args.batchsize,
                                               num_workers=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=source_collate_fn_tr_fda)
    test_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=args.batchsize,
                                                num_workers=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = UNet()
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    start_epoch = 0
    amp_grad_scaler = GradScaler()
    criterion_dice = DiceLoss()

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_seg_loss_list = list()
        train_com_loss_list = list()
        for iter, batch in enumerate(train_dataloader):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            weight = torch.from_numpy(batch['weight']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(normalize_image(batch['fda_data'])).cuda().to(dtype=torch.float32)
            optimizer.zero_grad()
            with autocast():
                sfs_3, sfs_2, sfs_1, sfs_0, global_features, decoder_features_0, decoder_features_1, decoder_features_2, decoder_features_3, output = model(data, mfeat=True)
                fda_sfs_3, fda_sfs_2, fda_sfs_1, fda_sfs_0, fda_global_features, fda_decoder_features_0, fda_decoder_features_1, fda_decoder_features_2, fda_decoder_features_3, fda_output = model(fda_data, mfeat=True)
                # encoder feature contrast loss
                compact_sfs_loss_3 = F.l1_loss(sfs_3, fda_sfs_3.detach(), reduction='mean') + \
                                      F.l1_loss(fda_sfs_3, sfs_3.detach(), reduction='mean')
                compact_sfs_loss_2 = F.l1_loss(sfs_2, fda_sfs_2.detach(), reduction='mean') + \
                                   F.l1_loss(fda_sfs_2, sfs_2.detach(), reduction='mean')
                compact_sfs_loss_1 = F.l1_loss(sfs_1, fda_sfs_1.detach(), reduction='mean') + \
                                   F.l1_loss(fda_sfs_1, sfs_1.detach(), reduction='mean')
                compact_sfs_loss_0 = F.l1_loss(sfs_0, fda_sfs_0.detach(), reduction='mean') + \
                                   F.l1_loss(fda_sfs_0, sfs_0.detach(), reduction='mean')
                compact_sfs_loss = compact_sfs_loss_0 + compact_sfs_loss_1 + compact_sfs_loss_2 + compact_sfs_loss_3
                # decoder feature contrast loss
                compact_decoder_loss_0 = F.l1_loss(decoder_features_0, fda_decoder_features_0.detach(), reduction='mean') + \
                                      F.l1_loss(fda_decoder_features_0, decoder_features_0.detach(), reduction='mean')
                compact_decoder_loss_1 = F.l1_loss(decoder_features_1, fda_decoder_features_1.detach(), reduction='mean') + \
                                         F.l1_loss(fda_decoder_features_1, decoder_features_1.detach(), reduction='mean')
                compact_decoder_loss_2 = F.l1_loss(decoder_features_2, fda_decoder_features_2.detach(), reduction='mean') + \
                                         F.l1_loss(fda_decoder_features_2, decoder_features_2.detach(), reduction='mean')
                compact_decoder_loss_3 = F.l1_loss(decoder_features_3, fda_decoder_features_3.detach(), reduction='mean') + \
                                         F.l1_loss(fda_decoder_features_3, decoder_features_3.detach(), reduction='mean')
                compact_decoder_loss = compact_decoder_loss_0 + compact_decoder_loss_1 + compact_decoder_loss_2 + compact_decoder_loss_3
                # global feature contrast loss
                compact_global_loss = F.l1_loss(fda_global_features, global_features.detach(), reduction='mean') + \
                               F.l1_loss(global_features, fda_global_features.detach(), reduction='mean')
                # full-scale feature-level contrast loss
                compact_loss = 0.5 * compact_global_loss + 0.1 * compact_decoder_loss + 0.4 * compact_sfs_loss

                sigmoid_output = torch.sigmoid(output)
                fda_sigmoid_output = torch.sigmoid(fda_output)
                # clinical prior-guided labellevel contrast loss
                seg_dice_loss = criterion_dice(sigmoid_output[:, 0], (seg[:, 0] > 0) * 1.0, weight=weight[:, 0]) + \
                           criterion_dice(sigmoid_output[:, 1], (seg[:, 0] == 2) * 1.0, weight=weight[:, 1]) + \
                           criterion_dice(fda_sigmoid_output[:, 0], (seg[:, 0] > 0) * 1.0, weight=weight[:, 0]) + \
                           criterion_dice(fda_sigmoid_output[:, 1], (seg[:, 0] == 2) * 1.0, weight=weight[:, 1])

                seg_smooth_loss = smooth_loss(sigmoid_output[:, 0], (seg[:, 0] > 0) * 1.0, weight=weight[:, 0]) + \
                           smooth_loss(sigmoid_output[:, 1], (seg[:, 0] == 2) * 1.0, weight=weight[:, 1]) + \
                           smooth_loss(fda_sigmoid_output[:, 0], (seg[:, 0] > 0) * 1.0, weight=weight[:, 0]) + \
                           smooth_loss(fda_sigmoid_output[:, 1], (seg[:, 0] == 2) * 1.0, weight=weight[:, 1])
                seg_loss = 0.9 * seg_dice_loss + 0.1 * seg_smooth_loss
                loss = seg_loss + 0.001 * compact_loss
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(loss.detach().cpu().numpy())
            train_seg_loss_list.append(seg_loss.detach().cpu().numpy())
            train_com_loss_list.append(compact_loss.detach().cpu().numpy())
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_seg_loss = np.mean(train_seg_loss_list)
        mean_tr_com_loss = np.mean(train_com_loss_list)
        print('  Tr loss: {}\n'
              '  Tr seg loss: {}; com loss: {}'.format(mean_tr_loss, mean_tr_seg_loss, mean_tr_com_loss))

        visualization_folder_epoch = os.path.join(visualization_folder+'/', str(epoch))
        os.makedirs(visualization_folder_epoch, exist_ok=True)
        output_list = list()
        name_list = list()
        with torch.no_grad():
            model.eval()
            for iter, batch in enumerate(test_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                name = batch['name']
                with autocast():
                    output = model(data)
                output_sigmoid = torch.sigmoid(output).cpu()
                for i in range(data.shape[0]):
                    case_seg = np.zeros((512, 512))
                    case_seg[output_sigmoid[i][0] > 0.5] = 255
                    case_seg[output_sigmoid[i][1] > 0.5] = 128
                    case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
                    case_seg_f.save(
                        os.path.join(visualization_folder_epoch, name[i].split('/')[-1].replace('.tif', '-1.tif')))
        eval_print_all_CU(visualization_folder_epoch + '/', os.path.join(root_folder, 'RIGA-mask', inference_tag, 'Labeled'))
        saved_model = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        print('  Saving model_{}.model...'.format('latest'))
        torch.save(saved_model, os.path.join(visualization_folder_epoch + '/', 'model.model'))