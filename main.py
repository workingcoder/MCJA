"""MCJA/main.py
   It is the main entry point for training the Multi-level Cross-modality Joint Alignment (MCJA) method.
"""

import os
import glob
import pprint
import logging

import numpy as np
import scipy.io as sio

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from apex import amp

from data import get_train_loader
from data import get_test_loader
from models.mcja import MCJA
from engine import get_trainer
from engine.engine import create_eval_engine
from utils.eval_data import eval_sysu, eval_regdb


def train(cfg):
    # Recorder ---------------------------------------------------------------------------------------------------------
    logger = logging.getLogger('MCJA')
    tb_dir = os.path.join(cfg.log_dir, 'tensorboard')
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Train DataLoader -------------------------------------------------------------------------------------------------
    train_loader = get_train_loader(dataset=cfg.dataset, root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    image_size=cfg.image_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    vimc_wg=cfg.vimc_wg,
                                    vimc_cc=cfg.vimc_cc,
                                    vimc_sj=cfg.vimc_sj,
                                    num_workers=4)

    # Test DataLoader --------------------------------------------------------------------------------------------------
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=cfg.batch_size,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)

    # Model ------------------------------------------------------------------------------------------------------------
    model = MCJA(num_classes=cfg.num_id,
                 drop_last_stride=cfg.drop_last_stride,
                 mda_ratio=cfg.mda_ratio,
                 mda_m=cfg.mda_m,
                 loss_id=cfg.loss_id,
                 loss_cmr=cfg.loss_cmr)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    logger.info(f'Model Parameter Num - {get_parameter_number(model)}')

    model.cuda()

    # Optimizer --------------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    model, optimizer = amp.initialize(model, optimizer, enabled=cfg.fp16, opt_level='O1', verbosity=0)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.lr_step, gamma=0.1)

    # Resume -----------------------------------------------------------------------------------------------------------
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        for key in list(checkpoint.keys()):
            model_state_dict = model.state_dict()
            if key in model_state_dict:
                if torch.is_tensor(checkpoint[key]) and checkpoint[key].shape != model_state_dict[key].shape:
                    logger.info(f'Warning during loading weights - Auto remove mismatch key: {key}')
                    checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=False)

    # Engine -----------------------------------------------------------------------------------------------------------
    checkpoint_dir = os.path.join('ckptlog/', cfg.dataset, cfg.prefix)
    engine = get_trainer(dataset=cfg.dataset,
                         model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         writer=writer,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         start_eval=cfg.start_eval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader)
    engine.run(train_loader, max_epochs=cfg.num_epoch)
    writer.close()


def test(cfg):
    # Logger -----------------------------------------------------------------------------------------------------------
    logger = logging.getLogger('MCJA')
    logger.info('\n## Starting the testing process...')

    # Test DataLoader --------------------------------------------------------------------------------------------------
    gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                   root=cfg.data_root,
                                                   batch_size=cfg.batch_size,
                                                   image_size=cfg.image_size,
                                                   num_workers=4,
                                                   mode=None)
    if cfg.mser:
        gallery_loader_r, query_loader_r = get_test_loader(dataset=cfg.dataset,
                                                           root=cfg.data_root,
                                                           batch_size=cfg.batch_size,
                                                           image_size=cfg.image_size,
                                                           num_workers=4,
                                                           mode='r')
        gallery_loader_g, query_loader_g = get_test_loader(dataset=cfg.dataset,
                                                           root=cfg.data_root,
                                                           batch_size=cfg.batch_size,
                                                           image_size=cfg.image_size,
                                                           num_workers=4,
                                                           mode='g')
        gallery_loader_b, query_loader_b = get_test_loader(dataset=cfg.dataset,
                                                           root=cfg.data_root,
                                                           batch_size=cfg.batch_size,
                                                           image_size=cfg.image_size,
                                                           num_workers=4,
                                                           mode='b')

    # Model ------------------------------------------------------------------------------------------------------------
    model = MCJA(num_classes=cfg.num_id,
                 drop_last_stride=cfg.drop_last_stride,
                 mda_ratio=cfg.mda_ratio,
                 mda_m=cfg.mda_m,
                 loss_id=cfg.loss_id,
                 loss_cmr=cfg.loss_cmr)
    model.cuda()
    model = amp.initialize(model, enabled=cfg.fp16, opt_level='O1', verbosity=0)

    # Resume -----------------------------------------------------------------------------------------------------------
    resume_path = cfg.resume if cfg.resume else glob.glob(f'{cfg.log_dir}/*best*')[0]
    ## Note: if cfg.resume is specified, it will be used;
    ## otherwise, the best model trained in the current experiment will be automatically loaded.
    checkpoint = torch.load(resume_path)
    for key in list(checkpoint.keys()):
        model_state_dict = model.state_dict()
        if key in model_state_dict:
            if torch.is_tensor(checkpoint[key]) and checkpoint[key].shape != model_state_dict[key].shape:
                logger.info(f'Warning during loading weights - Auto remove mismatch key: {key}')
                checkpoint.pop(key)
    model.load_state_dict(checkpoint, strict=False)

    # Evaluator --------------------------------------------------------------------------------------------------------
    non_blocking = True
    evaluator = create_eval_engine(model, non_blocking)
    # extract query feature
    evaluator.run(query_loader)
    q_feats = torch.cat(evaluator.state.feat_list, dim=0)
    q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
    # extract gallery feature
    evaluator.run(gallery_loader)
    g_feats = torch.cat(evaluator.state.feat_list, dim=0)
    g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
    g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

    if cfg.mser:
        ###### Multi-Spectral Enhanced Ranking (MSER) ######
        evaluator = create_eval_engine(model, non_blocking)
        # extract query feature  mode - r
        evaluator.run(query_loader_r)
        q_feats_r = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids_r = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams_r = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths_r = np.concatenate(evaluator.state.img_path_list, axis=0)
        # extract gallery feature  mode - r
        evaluator.run(gallery_loader_r)
        g_feats_r = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids_r = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams_r = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths_r = np.concatenate(evaluator.state.img_path_list, axis=0)

        evaluator = create_eval_engine(model, non_blocking)
        # extract query feature  mode - g
        evaluator.run(query_loader_g)
        q_feats_g = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids_g = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams_g = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths_g = np.concatenate(evaluator.state.img_path_list, axis=0)
        # extract gallery feature  mode - g
        evaluator.run(gallery_loader_g)
        g_feats_g = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids_g = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams_g = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths_g = np.concatenate(evaluator.state.img_path_list, axis=0)

        evaluator = create_eval_engine(model, non_blocking)
        # extract query feature  mode - b
        evaluator.run(query_loader_b)
        q_feats_b = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids_b = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams_b = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths_b = np.concatenate(evaluator.state.img_path_list, axis=0)
        # extract gallery feature  mode - b
        evaluator.run(gallery_loader_b)
        g_feats_b = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids_b = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams_b = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths_b = np.concatenate(evaluator.state.img_path_list, axis=0)

        q_feats_mser = [q_feats, q_feats_r, q_feats_g, q_feats_b]
        q_ids_mser = [q_ids, q_ids_r, q_ids_g, q_ids_b]
        q_cams_mser = [q_cams, q_cams_r, q_cams_g, q_cams_b]
        q_img_paths_mser = [q_img_paths, q_img_paths_r, q_img_paths_g, q_img_paths_b]
        g_feats_mser = [g_feats, g_feats_r, g_feats_g, g_feats_b]
        g_ids_mser = [g_ids, g_ids_r, g_ids_g, g_ids_b]
        g_cams_mser = [g_cams, g_cams_r, g_cams_g, g_cams_b]
        g_img_paths_mser = [g_img_paths, g_img_paths_r, g_img_paths_g, g_img_paths_b]

    if cfg.dataset == 'sysu':
        perm = sio.loadmat(os.path.join(cfg.data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']
        eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1)
        eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10)
        eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1)
        eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10)
        if cfg.mser:
            eval_sysu(q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser,
                      g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser,
                      perm, mode='all', num_shots=1, mser=True)
            eval_sysu(q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser,
                      g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser,
                      perm, mode='all', num_shots=10, mser=True)
            eval_sysu(q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser,
                      g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser,
                      perm, mode='indoor', num_shots=1, mser=True)
            eval_sysu(q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser,
                      g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser,
                      perm, mode='indoor', num_shots=10, mser=True)

    elif cfg.dataset == 'regdb':
        logger.info('Test Mode - infrared to visible')
        eval_regdb(q_feats, q_ids, q_cams, q_img_paths,
                   g_feats, g_ids, g_cams, g_img_paths, mode='i2v')
        logger.info('Test Mode - visible to infrared')
        eval_regdb(g_feats, g_ids, g_cams, g_img_paths,
                   q_feats, q_ids, q_cams, q_img_paths, mode='v2i')
        if cfg.mser:
            logger.info('Test Mode - infrared to visible')
            eval_regdb(q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser,
                       g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser, mode='i2v', mser=True)
            logger.info('Test Mode - visible to infrared')
            eval_regdb(g_feats_mser, g_ids_mser, g_cams_mser, g_img_paths_mser,
                       q_feats_mser, q_ids_mser, q_cams_mser, q_img_paths_mser, mode='v2i', mser=True)
    else:
        raise NotImplementedError(f'Dataset - {cfg.dataset} is not supported')

    evaluator.state.feat_list.clear()
    evaluator.state.id_list.clear()
    evaluator.state.cam_list.clear()
    evaluator.state.img_path_list.clear()


if __name__ == '__main__':
    # Tools ------------------------------------------------------------------------------------------------------------
    import argparse
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg
    from utils.tools import set_seed, time_str

    # Argument Parser --------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/SYSU_MCJA.yml', help='customized strategy config')
    parser.add_argument('--seed', type=int, default=8, help='random seed - choose a lucky number')
    parser.add_argument('--desc', type=str, default=None, help='auxiliary description of this experiment')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device for the training process')
    args = parser.parse_args()

    # Environment ------------------------------------------------------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    # Configuration ----------------------------------------------------------------------------------------------------
    ## strategy_cfg
    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)
    ## dataset_cfg
    dataset_cfg = dataset_cfg.get(cfg.dataset)
    for k, v in dataset_cfg.items():
        cfg[k] = v
    ## other cfg
    cfg.prefix += f'_Time-{time_str()}'
    cfg.prefix += f'_{args.desc}' if (args.desc is not None) else ''
    cfg['log_dir'] = os.path.join('ckptlog/', cfg.dataset, cfg.prefix)
    ## freeze cfg
    cfg.freeze()

    # Logger ---------------------------------------------------------------------------------------------------------
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir, exist_ok=True)
    logger = logging.getLogger('MCJA')
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(filename=os.path.join(cfg.log_dir, 'log.txt'))
    fileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    logger.info('\n' + pprint.pformat(cfg))

    # Train & Test -----------------------------------------------------------------------------------------------------
    if not cfg.test_only:
        train(cfg)
    test(cfg)

    # ------------------------------------------------------------------------------------------------------------------
