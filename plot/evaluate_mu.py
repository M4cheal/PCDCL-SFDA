import cv2
import numpy as np
import os
from plot.evaluation_metrics_for_segmentation import *
import xlwt
def return_list(data_path,data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

def eval_print_all_CU(pred_img_path, gt_img_path):
    f = open(pred_img_path + "_preCDR.txt", "a")
    new_workbook = xlwt.Workbook()
    sheet = new_workbook.add_sheet('1')

    file_list = return_list(pred_img_path, '.tif')
    n = len(file_list)
    DC_score_cup = {'img_name':'DC_cup score'}
    DC_score_disc = {'img_name':'DC_disc score'}
    JAC_score_cup = {'img_name':'JAC_cup score'}
    JAC_score_disc = {'img_name':'JAC_disc score'}
    ACC_score_cup = {'img_name':'ACC_cup score'}
    ACC_score_disc = {'img_name':'ACC_disc score'}
    SEN_score_cup = {'img_name':'SEN_cup score'}
    SEN_score_disc = {'img_name':'SEC_disc score'}
    SPC_score_cup = {'img_name':'SPC_cup score'}
    SPC_score_disc = {'img_name':'SPC_disc score'}
    CDR_score = {'img_name':'CDR score'}

    cup_dices = []
    disc_dices = []
    cup_JAC = []
    disc_JAC = []
    cup_ACC = []
    disc_ACC = []
    cup_SEN = []
    disc_SEN = []
    cup_SPC = []
    disc_SPC = []
    CDR = []

    for i in range(n):
        i = i
        temp_list = file_list[i]
        pred_name = os.path.join(pred_img_path,temp_list[:-4]+'.tif')
        gt_name = os.path.join(gt_img_path,temp_list[:-4]+'.tif')
        pred = cv2.resize(cv2.imread(pred_name,0),(512,512), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(cv2.imread(gt_name,0),(512,512), interpolation=cv2.INTER_NEAREST)
         
        gt_oc = cv2.resize(cv2.imread(gt_name, 0),(512,512), interpolation=cv2.INTER_NEAREST)
        gt_oc[gt_oc == 0] = 255
        gt_oc[gt_oc == 128] = 0

        gt_od = cv2.resize(cv2.imread(gt_name, 0),(512,512), interpolation=cv2.INTER_NEAREST)
        gt_od[gt_od == 128] = 255
        gt_od[gt_od == 0] = 1
        gt_od[gt_od == 255] = 0
        gt_od[gt_od == 1] = 255

        ret_oc, thresh_oc = cv2.threshold(gt_oc, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        ret_od, thresh_od = cv2.threshold(gt_od, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        contours_oc, hierarchy_oc = cv2.findContours(
            thresh_oc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_oc = contours_oc

        contours_od, hierarchy_od = cv2.findContours(
            thresh_od, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_od = contours_od
        pred_RGB = cv2.resize(cv2.imread(pred_name),(512,512), interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(pred_RGB, cnt_oc, -1, (0, 0, 255), 3)
        cv2.drawContours(pred_RGB, cnt_od, -1, (0, 255, 0), 3)

        cv2.imwrite(os.path.join(pred_img_path, 'TRUE_' + temp_list[:-4] + '.png'), pred_RGB)
         
        pred[pred == 0] = 1
        pred[pred == 128] = 0
        pred[pred == 255] = 128
        pred[pred == 1] = 255

        gt[gt == 0] = 1
        gt[gt == 128] = 0
        gt[gt == 255] = 128
        gt[gt == 1] = 255

        cup_dice,cup_jac,cup_acc,cup_sen,cup_spc,disc_dice,disc_jac,disc_acc,disc_sen,disc_spc,cdr, p_cdr = evaluate_binary_segmentation_CDR_ROC(pred,gt)
        DC_score_cup[temp_list] = cup_dice
        DC_score_disc[temp_list] = disc_dice
        JAC_score_cup[temp_list] = cup_jac
        JAC_score_disc[temp_list] = disc_jac
        ACC_score_cup[temp_list] = cup_acc
        ACC_score_disc[temp_list] = disc_acc
        SEN_score_cup[temp_list] = cup_sen
        SEN_score_disc[temp_list] = disc_sen
        SPC_score_cup[temp_list] = cup_spc
        SPC_score_disc[temp_list] = disc_spc
        CDR_score[temp_list] = cdr

        cup_dices.append(cup_dice)
        disc_dices.append(disc_dice)
        cup_JAC.append(cup_jac)
        disc_JAC.append(disc_jac)
        cup_ACC.append(cup_acc)
        disc_ACC.append(disc_acc)
        cup_SEN.append(cup_sen)
        disc_SEN.append(disc_sen)
        cup_SPC.append(cup_spc)
        disc_SPC.append(disc_spc)
        CDR.append(cdr)
        # print('!!!!!!!! Name :' + str(temp_list))
        # print('DISC_DICE :' + str(disc_dice))
        # print('CUP_DICE :' + str(cup_dice))
        # print('CDR mean :' + str(cdr))
        sheet.write(i+1, 1, str(temp_list))
        sheet.write(i+1, 2, str(disc_dice))
        sheet.write(i+1, 3, str(cup_dice))
        sheet.write(i+1, 4, str(cdr))
        f.write(str(temp_list) + '/' + str(p_cdr) + '\n')
    f.close()
    new_workbook.save(pred_img_path + "_metrics.xls")
    mean_cup_dice = np.mean(cup_dices)
    mean_disc_dice = np.mean(disc_dices)
    DC_score_cup[' DC_cup mean_score'] = mean_cup_dice
    DC_score_disc['DC_disc mean_score'] = mean_disc_dice
    mean_cup_jac = np.mean(cup_JAC)
    mean_disc_jac = np.mean(disc_JAC)
    JAC_score_cup[' JAC_cup mean_score'] = mean_cup_jac
    JAC_score_disc['JAC_disc mean_score'] = mean_disc_jac
    mean_cup_acc = np.mean(cup_ACC)
    mean_disc_acc = np.mean(disc_ACC)
    ACC_score_cup[' ACC_cup mean_score'] = mean_cup_acc
    ACC_score_disc['ACC_disc mean_score'] = mean_disc_acc
    mean_cup_sen = np.mean(cup_SEN)
    mean_disc_sen = np.mean(disc_SEN)
    SEN_score_cup[' SEM_cup mean_score'] = mean_cup_sen
    SEN_score_disc['SEN_disc mean_score'] = mean_disc_sen
    mean_cup_spc = np.mean(cup_SPC)
    mean_disc_spc = np.mean(disc_SPC)
    SPC_score_cup[' SPC_cup mean_score'] = mean_cup_spc
    SPC_score_disc['SPC_disc mean_score'] = mean_disc_spc

    mean_cdr = np.mean(CDR)
    CDR_score[' CDR mean_score'] = mean_cdr

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('DISC_DICE mean :'+str(mean_disc_dice))
    print('CUP_DICE mean :'+str(mean_cup_dice))
    print('DISC_JAC mean :'+str(mean_disc_jac))
    print('CUP_JAC mean :'+str(mean_cup_jac))
    print('DISC_ACC mean :'+str(mean_disc_acc))
    print('CUP_ACC mean :'+str(mean_cup_acc))
    print('DISC_SEN mean :'+str(mean_disc_sen))
    print('CUP_SEN mean :'+str(mean_cup_sen))
    print('DISC_SPC mean :'+str(mean_disc_spc))
    print('CUP_SPC mean :'+str(mean_cup_spc))
    print('CDR mean :'+str(mean_cdr))
    f1 = open(pred_img_path + "_result.txt", "a")
    f1.write('DISC_DICE mean :' + str(mean_disc_dice) + '\n')
    f1.write('CUP_DICE mean :' + str(mean_cup_dice) + '\n')
    f1.write('DISC_JAC mean :' + str(mean_disc_jac) + '\n')
    f1.write('CUP_JAC mean :' + str(mean_cup_jac) + '\n')
    f1.write('DISC_ACC mean :' + str(mean_disc_acc) + '\n')
    f1.write('CUP_ACC mean :' + str(mean_cup_acc) + '\n')
    f1.write('DISC_SEN mean :' + str(mean_disc_sen) + '\n')
    f1.write('CUP_SEN mean :' + str(mean_cup_sen) + '\n')
    f1.write('DISC_SPC mean :' + str(mean_disc_spc) + '\n')
    f1.write('CUP_SPC mean :' + str(mean_cup_spc) + '\n')
    f1.write('CDR mean :' + str(mean_cdr) + '\n')
    f1.close()
