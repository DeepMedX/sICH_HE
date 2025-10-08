import os 
import argparse 
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler 
import classic_models 
from utils.lr_methods import warmup 
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate 

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
parser.add_argument('--epochs', type=int, default=100, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate') 
parser.add_argument('--seed', default=21, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision') 
parser.add_argument('--data_path', type=str, default=r"data")
parser.add_argument('--model', type=str, default="vision_transformer", help=' select a model for training') 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--weights', type=str, default=r'model_pth/vit_base_patch16_224.pth', help='initial weights path')
# parser.add_argument('--weights', type=str, default=r'model_pth/resnet101.pth', help='initial weights path')

opt = parser.parse_args()  

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed) # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed) # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        print('random seed has been fixed')
    seed_torch() 

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if opt.tensorboard:
        # 这是存放你要使用tensorboard显示的数据的绝对路径
        log_path = os.path.join('./results/tensorboard' , args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil 

        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])} 
   
    train_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'train'), transform=data_transform["train"])
    val_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'val'), transform=data_transform["val"]) 
 
    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))
    
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,  num_workers=0, collate_fn=val_dataset.collate_fn)
 
    # create model
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device) 
    
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        
        # # Resnet权重加载
        # weights_dict = torch.load(args.weights, map_location=torch.device('cpu'))
        # # # weights_dict = {k: v.to('cuda:0') for k, v in weights_dict.items()}  # 将权重转移到GPU
        # del weights_dict['fc.weight']
        # del weights_dict['fc.bias']

        # # 删除不需要的权重
        #del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #   else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        # VIT权重加载
        weights_dict = torch.load(args.weights, map_location=torch.device('cuda'))
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        
        print(model.load_state_dict(weights_dict, strict=False))
        
    pg = [p for p in model.parameters() if p.requires_grad] 
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.
    
    # save parameters path
    save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, use_amp=args.use_amp, lr_method= warmup)
        scheduler.step()
        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)

 
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc))   
        with open(os.path.join(save_path, "vit.txt"), 'a') as f: 
                f.writelines('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc) + '\n')

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, "vit.pth")) 

        
main(opt)

