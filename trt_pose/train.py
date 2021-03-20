import argparse
import subprocess
import torch
import torchvision
import os
import torch.optim
import tqdm
import apex.amp as amp
import time
import json
import pprint
import torch.nn.functional as F
from trt_pose.coco import CocoDataset, CocoHumanPoseEval
from trt_pose.models import MODELS

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6

def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, epoch):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'epoch_%d.pth' % epoch)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, test_loss):
    with open(logfile, 'a+') as f:
        logline = '%d, %f, %f' % (epoch, train_loss, test_loss)
        print(logline)
        f.write(logline + '\n')
        
device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    logfile_path = args.config + '.log'
    
    checkpoint_dir = args.config + '.checkpoints'
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)
    
        
    # LOAD DATASETS
    
    train_dataset_kwargs = config["train_dataset"]
    train_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(**config['color_jitter']),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset_kwargs = config["test_dataset"]
    test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if 'evaluation' in config:
        evaluator = CocoHumanPoseEval(**config['evaluation'])
    
    train_dataset = CocoDataset(**train_dataset_kwargs)
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    part_type_counts = test_dataset.get_part_type_counts().float().cuda()
    part_weight = 1.0 / part_type_counts
    part_weight = part_weight / torch.sum(part_weight)
    # paf_type_counts = test_dataset.get_paf_type_counts().float().cuda()
    # paf_weight = 1.0 / paf_type_counts
    # paf_weight = paf_weight / torch.sum(paf_weight)
    # paf_weight /= 2.0
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    
    if "initial_state_dict" in config['model']:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))
    
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if 'mask_unlabeled' in config and config['mask_unlabeled']:
        print('Masking unlabeled annotations')
        mask_unlabeled = True
    else:
        mask_unlabeled = False
        
    for epoch in range(config["epochs"]):
        if str(epoch) in config['stdev_schedule']:
            stdev = config['stdev_schedule'][str(epoch)]
            print('Adjusting stdev to %f' % stdev)
            train_dataset.stdev = stdev
            test_dataset.stdev = stdev
            
        if str(epoch) in config['lr_schedule']:
            new_lr = config['lr_schedule'][str(epoch)]
            print('Adjusting learning rate to %f' % new_lr)
            set_lr(optimizer, new_lr)
        
        if epoch % config['checkpoints']['interval'] == 0:
            save_checkpoint(model, checkpoint_dir, epoch)
                
        train_loss = 0.0
        model = model.train()
        number_of_train_instance = 0
        for images, cmaps, masks in tqdm.tqdm(iter(train_loader)):
            for i in range(len(images)):
                number_of_instance += len(images)
                image = images[i].to(device)
                cmap = cmap[i].to(device)
                
                if mask_unlabeled:
                    mask = masks[i].to(device).float()
                else:
                    mask = torch.ones_like(masks[i]).to(device).float()
                
                optimizer.zero_grad()
                cmap_out = model(image)
                
                cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
                
                loss = cmap_mse
                
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                train_loss += float(loss)
        
        train_loss /= number_of_train_instance
        
        test_loss = 0.0
        model = model.eval()
        number_of_val_instance = 0
        for images, cmaps, masks in tqdm.tqdm(iter(test_loader)):
            for j in len(images):
                with torch.no_grad():
                    image = images[i].to(device)
                    cmap = cmaps[i].to(device)
                    mask = masks[i].to(device).float()

                    if mask_unlabeled:
                        mask = masks[i].to(device).float()
                    else:
                        mask = torch.ones_like(masks[i]).to(device).float()
                    
                    cmap_out = model(image)
                    
                    cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)

                    loss = cmap_mse

                    test_loss += float(loss)
        test_loss /= number_of_val_instance
        
        write_log_entry(logfile_path, epoch, train_loss, test_loss)
        
        if 'evaluation' in config:
            evaluator.evaluate(model, train_dataset.topology)