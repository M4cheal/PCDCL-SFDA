import argparse
import os
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloaders.transform import collate_fn_tr, collate_fn_ts
from dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from dataloaders.convert_csv_to_list import convert_labeled_list

from PIL import Image
from sklearn.cluster import KMeans
from networks.unet import UNet
from plot.evaluate_mu import eval_print_all_CU
from utils.util import sigmoid_entropy

if __name__ == '__main__':
    target_dataset = "MESSIDOR_Base1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str,
                        default=r'D:\SFDA\ProSFDA-master\prosfda\log\UNet_Source_Model\checkpoints\model_final.model')
    parser.add_argument('--batchsize', type=int, default=1)  # 1
    parser.add_argument('-g', '--gpu', type=int, default=3)
    parser.add_argument('--root', default=r'D:\RIGAPlus')
    parser.add_argument('--tr_csv', help='training csv file.', default=[r'D:\RIGAPlus\{}_unlabeled.csv'.format(target_dataset)])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    inference_tag = target_dataset
    visualization_folder = r'D:\SFDA\PCDCL-SFDA\logs\pseudo\{}\Unlabeled'.format(target_dataset)
    os.makedirs(visualization_folder, exist_ok=True)
    tr_csv = tuple(args.tr_csv)
    root_folder = args.root
    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_unlabeled_set(root_folder, tr_img_list)
    train_loader = torch.utils.data.DataLoader(tr_dataset,
                                               batch_size=args.batchsize,
                                               num_workers=1,
                                               shuffle=False,
                                               pin_memory=True,
                                               collate_fn=collate_fn_tr)

    model = UNet()
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    WkG = checkpoint['model_state_dict']['up5.weight']

    model.eval()
    pseudo_label_name = []
    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, img_name = sample['data'], sample['name']
            data = torch.from_numpy(data).to(dtype=torch.float32)
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)

            with torch.no_grad():
                preds, feature = model(data, rfeat=True)
            prediction_sigmoid = torch.sigmoid(preds)
            pseudo_label_s = prediction_sigmoid.clone()
            pseudo_label_s[pseudo_label_s > 0.5] = 1.0
            pseudo_label_s[pseudo_label_s <= 0.5] = 0.0

            prediction_entropy = prediction_sigmoid.clone()

            prediction_entropy = sigmoid_entropy(prediction_entropy)
            prediction_entropy[torch.isnan(prediction_entropy)] = 0
            prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
                        1 / (prediction_entropy.max() - prediction_entropy.min())))
            high_entropy = prediction_entropy.clone()
            high_entropy[high_entropy > 0.5] = 1
            high_entropy[high_entropy <= 0.5] = 0

            pseudo_label = pseudo_label_s.int() | high_entropy.int()
            pseudo_label = pseudo_label.float()

            mask_OD_obj = F.interpolate(pseudo_label_s[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
            mask_OC_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            mask_OC_bck = 1.0 - mask_OC_obj

            prediction_sigmoid_r = F.interpolate(prediction_sigmoid, size=feature.size()[2:], mode='bilinear',
                                                 align_corners=True)
            prediction_entropy_r = F.interpolate(prediction_entropy, size=feature.size()[2:], mode='bilinear',
                                                 align_corners=True)

            feature_OC_obj = feature * prediction_entropy_r[:, 1:, ...] + feature * prediction_sigmoid_r[:, 1:, ...]

            centroid_OC = torch.mean(WkG.transpose(0, 1), dim=[2, 3], keepdim=True)[1]
            centroid_OC_bck = torch.sum(feature_OC_obj * mask_OC_bck, dim=[0, 2, 3], keepdim=True) / torch.sum(
                mask_OC_bck, dim=[0, 2, 3], keepdim=True)

            feature_OC_obj = feature_OC_obj.cpu().numpy().reshape(256, 256 * 256).transpose(1, 0)
            kmeans_init_OC = torch.cat([centroid_OC_bck.squeeze(-1).squeeze(0), centroid_OC.squeeze(-1).squeeze(0)],
                                       dim=1).transpose(0, 1).cpu().numpy()
            kmeans_OC = KMeans(n_clusters=2, random_state=0, init=kmeans_init_OC, n_init=1).fit(feature_OC_obj)
            labels_OC = kmeans_OC.labels_.reshape(256, 256)

            for i in range(prediction_sigmoid.shape[0]):
                pseudo_label_name.append(img_name[i].split('/')[-1])
                case_seg = np.zeros((256, 256))
                case_seg[mask_OD_obj.cpu()[i][0] == 1] = 255
                case_seg[labels_OC == 1] = 128
                case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
                case_seg_f.save(
                    os.path.join(visualization_folder, img_name[i].split('/')[-1].replace('.tif', '-1.tif')))
                
    # eval_print_all_CU(visualization_folder + '/', os.path.join(root_folder, 'RIGA-mask', inference_tag, 'Labeled'))
    with open(os.path.join(visualization_folder, 'generate_pseudo.csv'), 'w') as f:
        f.write('image,mask\n')
        for i in pseudo_label_name:
            f.write('RIGA/{}/Unlabeled/{},RIGA-pseudo-our/{}/Unlabeled/{}\n'.format(inference_tag, i, inference_tag, i))
