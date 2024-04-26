######################################
#         Kaihua Tang
######################################

import os

import torch
import torch.nn as nn
import torch.nn.functional as F 

import utils.general_utils as utils
from data.dataloader import get_loader

INVALID_SPLITS_DATASET = ('Animal10N', 'Food101N', "Animal10NLT", "Food101NLT")
INVALID_ATTRIBUTE_DATASET = ('MSCOCO-LT', 'MSCOCO-BL', 'Places-GLT', 'Animal10N', 'Food101N', "Animal10NLT", "Food101NLT")

class acc_by_splits():
    def __init__(self, logger, stype):
        self.logger = logger
        self.stype = stype
    
    def print_score(self, predictions, labels, num_class, split, csv_results=None, premask=False):
        split_set = sorted(list(set(split.tolist())))
        self.logger.info('------------- Detailed Splits by {} -----------'.format(self.stype))

        output_log = 'Recall/AC ==> '
        for s in split_set:
            s_mask = (split==s)
            if premask:
                s_acc = utils.calculate_recall(predictions[s_mask].view(-1), labels[s_mask].view(-1))  
            else:
                s_acc = utils.calculate_recall(predictions.view(-1), labels.view(-1), split_mask=s_mask)  
            output_log = output_log + '{}_{} Acc : {:7.4f} / {:5d}, '.format(self.stype, str(s), s_acc, s_mask.sum().item())
            if csv_results is not None:
                csv_results['recall'].append('{:.4f}'.format(s_acc))
        self.logger.info(output_log)

        output_log = 'Precision ==> '
        for s in split_set:
            s_mask = (split==s)
            if premask:
                s_prc = utils.calculate_precision(predictions[s_mask].view(-1), labels[s_mask].view(-1), num_class)
            else:
                s_prc = utils.calculate_precision(predictions.view(-1), labels.view(-1), num_class, split_mask=s_mask)
            output_log = output_log + '{}_{} Prc : {:7.4f} / {:5d}, '.format(self.stype, str(s), s_prc, s_mask.sum().item())
            if csv_results is not None:
                csv_results['precision'].append('{:.4f}'.format(s_prc))
        self.logger.info(output_log)

class test_baseline():
    def __init__(self, config, logger, model, classifier, val=False, specify_testset=None, add_ckpt=None):
        self.config = config
        self.logger = logger
        self.model = model
        self.classifier = classifier
        self.add_ckpt = add_ckpt

        self.save_all = self.config['saving_opt']['save_all']

        # get dataloader
        if val:
            self.phase = 'val'
            self.loader = get_loader(config, 'val', config['dataset']['testset'], logger)
        elif specify_testset is not None:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', specify_testset, logger)
        else:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', config['dataset']['testset'], logger)


    def run_test(self):
        # set model to evaluation
        self.model.eval()
        self.classifier.eval()

        if self.config['dataset']['name'] not in INVALID_SPLITS_DATASET:
            currect_split = self.config['dataset']['testset']
            self.logger.info('------------- Start Testing at Split: {} -----------'.format(currect_split))

            if currect_split in ('test_bl', 'test_bbl'):
                frq_accs = acc_by_splits(self.logger, 'frequency')
            if currect_split in ('test_bbl'):
                if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                    att_accs = acc_by_splits(self.logger, 'attribute')
            # run batch
            with torch.no_grad():
                all_preds = []
                all_labs = []
                all_frqs = []
                all_atts = []
                
                save_features = []
                save_preds = []
                save_labs = []
                save_frqs = []
                save_atts = []
                save_inds = []

                for _, (inputs, labels, freq_labels, attributes, indexes) in enumerate(self.loader):
                    
                    # additional inputs
                    inputs, labels, freq_labels, attributes = inputs.cuda(), labels.cuda(), freq_labels.cuda(), attributes.cuda()
                    add_inputs = {}
    
                    features = self.model(inputs)
                    predictions = self.classifier(features, add_inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    all_preds.append(predictions)
                    all_labs.append(labels)
                    all_frqs.append(freq_labels)
                    all_atts.append(attributes)

                    # save output
                    if self.save_all:
                        save_features.append(features.detach().clone().cpu())
                        save_preds.append(predictions.detach().clone().cpu())
                        save_labs.append(labels.detach().clone().cpu())
                        save_frqs.append(freq_labels.detach().clone().cpu())
                        save_atts.append(attributes.detach().clone().cpu())
                        save_inds.append(indexes.detach().clone().cpu())
                
                # save output
                if self.save_all:
                    self.logger.info('============ Start Saving Test Outputs ===============')
                    save_outputs = {'save_features': torch.cat(save_features, dim=0),
                                    'save_preds' : torch.cat(save_preds, dim=0),
                                    'save_labs' : torch.cat(save_labs, dim=0),
                                    'save_frqs' : torch.cat(save_frqs, dim=0),
                                    'save_atts' : torch.cat(save_atts, dim=0),
                                    'save_inds' : torch.cat(save_inds, dim=0), }
                    model_path = os.path.join(self.config['output_dir'], 'save_outputs_{}.pth'.format(self.config['dataset']['testset']))
                    torch.save(save_outputs, model_path)
                    self.logger.info('============ Test Outputs Saved to {} ==============='.format(model_path))


                all_preds = torch.cat(all_preds, dim=0)
                num_class = all_preds.shape[-1]
                all_preds = all_preds.max(-1)[1].view(-1)
                all_labs = torch.cat(all_labs, dim=0).view(-1)
                all_frqs = torch.cat(all_frqs, dim=0).view(-1)
                all_atts = torch.cat(all_atts, dim=0).view(-1)

                csv_results = {'recall': [currect_split, 'recall'], 
                                'precision': [currect_split, 'precision'], 
                                'f1': [currect_split, 'f1']}

                # frequency splits by intra-class attributes
                if currect_split in ('test_bl', 'test_bbl'):
                    frq_accs.print_score(all_preds, all_labs, num_class, split=all_frqs, csv_results=csv_results)
                if currect_split in ('test_bbl'):
                    if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                        att_accs.print_score(all_preds, all_labs, num_class, split=all_atts, csv_results=csv_results, premask=True)

                # overall performance
                total_num = all_preds.shape[0]
                # calculate recall
                recall_score = utils.calculate_recall(all_preds, all_labs)
                self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
                csv_results['recall'].append('{:.4f}'.format(recall_score))
                # calculate precision
                precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
                self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
                csv_results['precision'].append('{:.4f}'.format(precision_score))
                # F1 score
                F1_score = utils.calculate_f1(recall_score, precision_score)
                self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))
                csv_results['f1'].append('{:.4f}'.format(F1_score))

                # save csv results
                self.logger.write_results(csv_results['recall'])
                self.logger.write_results(csv_results['precision'])
                self.logger.write_results(csv_results['f1'])
        
        else:
            # run batch
            with torch.no_grad():
                all_preds = []
                all_labs = []
                all_indices = []
                all_feat = []
                
                save_features = []
                save_preds = []
                save_labs = []
                save_inds = []


                for _, (inputs, labels, _, _, indexes) in enumerate(self.loader):
                    # additional inputs
                    inputs, labels = inputs.cuda(), labels.cuda()
                    add_inputs = {}
    
                    features = self.model(inputs)
                    predictions = self.classifier(features, add_inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    all_preds.append(predictions.detach().clone().cpu())
                    all_labs.append(labels.detach().clone().cpu())
                    all_indices.append(indexes.detach().clone().cpu())
                    all_feat.append(features.detach().clone().cpu())

                    # save output
                    if self.save_all:
                        save_features.append(features.detach().clone().cpu())
                        save_preds.append(predictions.detach().clone().cpu())
                        save_labs.append(labels.detach().clone().cpu())
                        save_inds.append(indexes.detach().clone().cpu())
                
                # save output
                if self.save_all:
                    self.logger.info('============ Start Saving Test Outputs ===============')
                    save_outputs = {'save_features': torch.cat(save_features, dim=0),
                                    'save_preds' : torch.cat(save_preds, dim=0),
                                    'save_labs' : torch.cat(save_labs, dim=0),
                                    'save_inds' : torch.cat(save_inds, dim=0), }
                    model_path = os.path.join(self.config['output_dir'], 'save_outputs_{}.pth'.format(self.config['dataset']['testset']))
                    torch.save(save_outputs, model_path)
                    self.logger.info('============ Test Outputs Saved to {} ==============='.format(model_path))


                all_preds = torch.cat(all_preds, dim=0)
                num_class = all_preds.shape[-1]
                all_preds = all_preds.max(-1)[1].view(-1)
                all_labs = torch.cat(all_labs, dim=0).view(-1)
                all_indices = torch.cat(all_indices, dim=0).view(-1)
                all_feat = torch.cat(all_feat, dim=0)

                from collections import Counter
                result = Counter(all_labs.tolist())
                print("result:", result)

                selected_label = 5
                selected_label_indices = torch.where(all_labs==selected_label)[0]
                # noises_indices = []
                # for ind in selected_label_indices:
                #     noises_indices.append(self.loader.dataset.noises[ind])
                # noises_indices = torch.Tensor(noises_indices)
                # noises_indices = torch.where(noises_indices == 0)[0]

                # import random
                # noises_indices = random.sample(noises_indices.tolist(), 5)

                # noises_indices = selected_label_indices[noises_indices]
                selected_label_indices = selected_label_indices.tolist()

                selected_num = 13
                import random
                selected_label_indices = random.sample(selected_label_indices, selected_num)
                # selected_label_indices.extend(noises_indices.tolist())
                selected_label_indices = list(set(selected_label_indices))
                # for x in selected_label_indices:
                #     print(x, self.loader.dataset.img_paths[x])

                

                selected_label2 = 4
                selected_label_indices2 = torch.where(all_labs==selected_label2)[0]
                selected_label_indices2 = selected_label_indices2.tolist()
                selected_num2 = 2
                selected_label_indices2 = random.sample(selected_label_indices2, selected_num2)
                selected_label_indices2 = list(set(selected_label_indices2))
                selected_label_indices.extend(selected_label_indices2)

                selected_label3 = 1
                selected_label_indices3 = torch.where(all_labs==selected_label3)[0]
                selected_label_indices3 = selected_label_indices3.tolist()
                selected_num3 = 2
                selected_label_indices3 = random.sample(selected_label_indices3, selected_num3)
                selected_label_indices3 = list(set(selected_label_indices3))
                selected_label_indices.extend(selected_label_indices3)



                selected_feat = all_feat[selected_label_indices]
                from utils.clusting_utils import CoarseLeadingForest
                clf = CoarseLeadingForest(selected_feat.numpy().tolist(), min_dist_multiple=0.6, max_dist_multiple=3.6, standardization=False)
                paths, _ = clf.generate_path()
                from functools import reduce
                for path in paths:
                    img_paths = []
                    img_noises = []
                    path_flatten = reduce(lambda x, y: x+y, path)
                    for index in path_flatten:
                        index = selected_label_indices[index]
                        img_paths.append(self.loader.dataset.img_paths[index])
                        # img_noises.append(self.loader.dataset.noises[index])
                    self.logger.info("path:"+str(path))
                    self.logger.info("img_paths:" + str(img_paths))
                    # self.logger.info("img_noises" + str(img_noises))

                get_noises = clf.get_noises(detailed_paths=paths, min_size=40, start_depth=2, num_layer=1, density_percentile=30)
                self.logger.info("get_noises:"+str(get_noises))
                # real_noises = []
                # for noise in get_noises:
                #     noise_ind = selected_label_indices[noise]
                #     real_noises.append(self.loader.dataset.noises[noise_ind])
                # self.logger.info("real_noises:"+str(real_noises))

                # real_noises = torch.Tensor(real_noises)
                # get_real_noises = torch.where(real_noises==0)[0]
                # self.logger.info("num_get_noises:"+str(len(get_noises)))
                # self.logger.info("num_real_noises:"+str(len(noises_indices)))
                # self.logger.info("num_get_real_noises:"+str(len(get_real_noises)))

                # overall performance
                total_num = all_preds.shape[0]
                # calculate recall
                recall_score = utils.calculate_recall(all_preds, all_labs)
                self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
                # calculate precision
                precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
                self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
                # F1 score
                F1_score = utils.calculate_f1(recall_score, precision_score)
                self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))



        # set back to training mode again
        self.model.train() 
        self.classifier.train()
        return recall_score
        
    def run_val(self, epoch):
        self.logger.info('------------- Start Validation at Epoch {} -----------'.format(epoch))
        # set model to evaluation
        self.model.eval()
        self.classifier.eval()

        if self.config['dataset']['name'] not in INVALID_SPLITS_DATASET:
            frq_accs = acc_by_splits(self.logger, 'frequency')
            if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                att_accs = acc_by_splits(self.logger, 'attribute')
            # run batch
            with torch.no_grad():
                all_preds = []
                all_labs = []
                all_frqs = []
                all_atts = []

                for _, (inputs, labels, freq_labels, attributes, indexes) in enumerate(self.loader):
                    # additional inputs
                    inputs, labels, freq_labels, attributes = inputs.cuda(), labels.cuda(), freq_labels.cuda(), attributes.cuda()
                    add_inputs = {}
    
                    features = self.model(inputs)
                    predictions = self.classifier(features, add_inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    all_preds.append(predictions)
                    all_labs.append(labels)
                    all_frqs.append(freq_labels)
                    all_atts.append(attributes)

            
                all_preds = torch.cat(all_preds, dim=0)
                num_class = all_preds.shape[-1]
                all_preds = all_preds.max(-1)[1].view(-1)
                all_labs = torch.cat(all_labs, dim=0).view(-1)
                all_frqs = torch.cat(all_frqs, dim=0).view(-1)
                all_atts = torch.cat(all_atts, dim=0).view(-1)

                # overall performance
                total_num = all_preds.shape[0]
                # calculate recall
                recall_score = utils.calculate_recall(all_preds, all_labs)
                self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
                # calculate precision
                precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
                self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
                # F1 score
                F1_score = utils.calculate_f1(recall_score, precision_score)
                self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))

                # frequency splits by intra-class attributes
                frq_accs.print_score(all_preds, all_labs, num_class, split=all_frqs)
                if self.config['dataset']['name'] not in INVALID_ATTRIBUTE_DATASET:
                    att_accs.print_score(all_preds, all_labs, num_class, split=all_atts, premask=True)
        else:
            # run batch
            with torch.no_grad():
                all_preds = []
                all_labs = []

                for _, (inputs, labels, _, _, _) in enumerate(self.loader):
                    # additional inputs
                    inputs, labels = inputs.cuda(), labels.cuda()
                    add_inputs = {}
    
                    features = self.model(inputs)
                    predictions = self.classifier(features, add_inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    all_preds.append(predictions)
                    all_labs.append(labels)

            
                all_preds = torch.cat(all_preds, dim=0)
                num_class = all_preds.shape[-1]
                all_preds = all_preds.max(-1)[1].view(-1)
                all_labs = torch.cat(all_labs, dim=0).view(-1)

                # overall performance
                total_num = all_preds.shape[0]
                # calculate recall
                recall_score = utils.calculate_recall(all_preds, all_labs)
                self.logger.info('Test Complete ==> Overall Recall/AC : {:9.4f}, Number Samples : {:9d}'.format(recall_score, total_num))
                # calculate precision
                precision_score = utils.calculate_precision(all_preds, all_labs, num_class)
                self.logger.info('Test Complete ==> Overall Precision : {:9.4f}, Number Samples : {:9d}'.format(precision_score, total_num))
                # F1 score
                F1_score = utils.calculate_f1(recall_score, precision_score)
                self.logger.info('Test Complete ==> Overall  F1 Score : {:9.4f}, Number Samples : {:9d}'.format(F1_score, total_num))


        # set back to training mode again
        self.model.train()
        self.classifier.train()
        return recall_score