'''
Author: Szl
Date: 2024-04-21 12:52:37
LastEditors: Szl
LastEditTime: 2024-04-22 23:40:42
Description: 
'''
# Package and Modules import statements
# -----------------------

import os
import random
import numpy as np
import torch
import torch.optim as optim # 导入需要的优化器对象
import torch.nn as nn

from models import Net
from config import HP
from dataloader import HandGestureDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser # 整个对象允许在通过命令行的方式启动python脚本时，运行额外在命令行中传递一些参数 - python3 xxx.py
from tensorboardX import SummaryWriter # tensorboardX是一个专用于机器学习模型训练成果的可视化工具，会通过读取记录下来的训练日志来可视化完整的loss训练下降过程
# SummaryWriter可以实时记录训练日志并以行缓冲的方式实时向日志文件中写入日志数据

# codings
# -----------------------

logger = SummaryWriter('./log')

# seed init: Ensure Reproducible Result
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)

def evaluate(model_, devloader, crit):
  model_.eval() # set evaluation flag
  sum_loss = 0.
  with torch.no_grad():
    for batch in devloader:
      x, y = batch
      pred = model_(x)
      loss = crit(pred, y.to(HP.device))
      sum_loss += loss.item()

  model_.train() # back to training mode
  return sum_loss / len(devloader)

def save_checkpoint(model_, epoch_, optm, checkpoint_path):
  save_dict = {
    'epoch': epoch_,
    'model_state_dict': model_.state_dict(),
    'optimizer_state_dict': optm.state_dict()
  }
  torch.save(save_dict, checkpoint_path)

def train():
  parser = ArgumentParser(description="Model Training")
  parser.add_argument(
      '--c',
      default=None,
      type=str,
      help='train from scratch or resume training'
  )
  args = parser.parse_args()

  # new model instance
  model = Net()
  model = model.to(HP.device)

  # loss function (loss.py)
  criterion = nn.CrossEntropyLoss()

  # optimizer
  opt = optim.Adam(model.parameters(), lr=HP.init_lr)
  # opt = optim.SGD(model.parameters(), lr=HP.init_lr)

  # train dataloader
  trainset = HandGestureDataset(HP.metadata_train_path)
  train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True)

  # dev datalader(evaluation)
  devset = HandGestureDataset(HP.metadata_eval_path)
  dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=False)

  start_epoch, step = 0, 0

  if args.c:
    checkpoint = torch.load(args.c)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Resume From %s.' % args.c)
  else:
    print('Training From scratch!')

  model.train()   # set training flag

  # main loop
  for epoch in range(start_epoch, HP.epochs):
    print('Start Epoch: %d, Steps: %d' % (epoch, len(train_loader)/HP.batch_size))
    for batch in train_loader:
      x, y = batch    # load data
      opt.zero_grad() # gradient clean
      pred = model(x) # forward process
      loss = criterion(pred, y.to(HP.device))   # loss calc

      loss.backward() # backward process
      opt.step()

      logger.add_scalar('Loss/Train', loss, step)

      if not step % HP.verbose_step:  # evaluate log print
        eval_loss = evaluate(model, dev_loader, criterion)
        logger.add_scalar('Loss/Dev', eval_loss, step)

      if not step % HP.verbose_step: # model save
        model_path = 'model_%d_%d.pth' % (epoch, step)
        save_checkpoint(model, epoch, opt, os.path.join('model', model_path))

      step += 1
      logger.flush()
      print('Epoch: [%d/%d], step: %d Train Loss: %.5f, Dev Loss: %.5f' % (epoch, HP.epochs, step, loss.item(), eval_loss))
  logger.close()

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  train()