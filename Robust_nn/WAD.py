import numpy as np
import matplotlib.pyplot as plt
from utils.utils import pgd
from utils.utils import progress_bar
import os
import time
import copy
import torch
import torchvision
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.distributions import uniform



class WAD2scale():
    def __init__(self, net_list, trainloader, testloader, 
                    device = 'cuda', 
                    num_adverse = 6,
                    scale_factor = 10,
                    kappa= None, 
                    eta = None, 
                    criterion = None,
                    adv_penalty = None,
                    penalty_coef = 3,
                    sd_perturbation_0 = 1,
                    path='./checkpoint'):
        '''
        WAD2scale: 
        ================================================
        Arguments:

        net_list: List of PyTorch nn network structures
        trainloader: PyTorch Dataloader
        testloader: PyTorch Dataloader
        device: 'cpu' or 'cuda' if GPU available
            type of decide to move tensors



        path: string
            path to save the best model
        
        '''      

        if not torch.cuda.is_available() and device=='cuda':
            raise ValueError("cuda is not available")

        self.net_list = []
        for net in net_list:
            self.net_list.append(net.to(device))
        
        self.device = device
        self.num_adverse = num_adverse
        self.trainloader, self.testloader = trainloader, testloader
        self.path = path
        self.scale_factor = scale_factor
        if kappa == None:
            self.kappa = { 'param': 0.1, 'adv': 0.1   }
        else:
            self.kappa = kappa
        if eta == None:
            self.eta = {'param': lambda t: 0.1/(t+1), 'adv': lambda t : 1/(t+1)}
        else:
            self.eta = eta   
        if criterion==None: 
           self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        if adv_penalty==None:
            self.adv_penalty = nn.MSELoss()
        else:
            self.adv_penalty = adv_penalty

        self.penalty_coef = penalty_coef
        self.sd_perturbation_0 =  sd_perturbation_0
        self.n_learners = len(net_list)
        self.test_acc_adv_best = 0
        self.train_loss, self.train_acc, self.train_reg = [], [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean, self.test_reg = [], [], [], []    
        self.train_times=[]
        self._restart_weights()


    def _restart_weights(self):
        self.net_weights = torch.ones(self.n_learners )/self.n_learners 
        self.adv_weights = torch.ones(self.num_adverse)/self.num_adverse

    def set_optimizer(self, optim_alg='Adam', args={'lr':1e-4}, scheduler=None, args_scheduler={}):
        '''
        Setting the optimizer of the list of network. The same optimizer and schedule is assumed
        ================================================
        Arguments:
        
        optim_alg : string
            Name of the optimizer
        args: dict
            Parameter of the optimizer
        scheduler: optim.lr_scheduler
            Learning rate scheduler
        args_scheduler : dict
            Parameters of the scheduler
        '''

        self.optim_list = []
        self.sched_list = []
        for i,net in enumerate(self.net_list):
            self.optim_list.append(getattr(optim, optim_alg)(net.parameters(), **args))
            if not scheduler:
                self.sched_list.append( optim.lr_scheduler.StepLR(self.optim_list[i], step_size=10**6, gamma=1))
            else:
                self.sched_list.append(getattr(optim.lr_scheduler, scheduler)(self.optim_list[i], **args_scheduler))

    def _optim_zeros(self):
        for optim in self.optim_list:
            optim.zero_grad()

    def _optim_step(self):
        for optim in self.optim_list:
            optim.step()

    def _sched_step(self):
        for  sched in self.sched_list:
            sched.step()


    # Voy aquí

    def train(self, epochs = 15):
        '''
        Training the network
        ================================================
        Arguments:

        epochs : int
            Number of epochs
        '''
        self.epochs = epochs
        for epoch in range(epochs):
            start_time = time.time()
            self._train(epoch)
            self.test(epoch)
            self.scheduler.step()
            self.train_times.append(time.time()-start_time)
        
    def _train(self, epoch):
        '''
        Training the model 
        '''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        reg, reg_term_1, norm_grad_sum  = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            #   Load inputs and target and 
            #   create copies of inputs and target (extend one rank to tensor)
            inputs_lst = torch.tile(inputs, (self.num_adverse,1,1 ,1,1))
            # targets_lst = torch.tile(targets, (self.num_adverse,1))
            inputs_lst, targets = inputs_lst.to(self.device), targets.to(self.device)
            print('Input list shape:',inputs_lst.shape, 
                    # 'Target_list shape:',targets_lst.shape,
                    'Inputs shape:',inputs.shape,
                    'tagets shape:',targets.shape)
            # create perturbed inputs
            pert_inputs = inputs_lst + torch.randn_like(inputs_lst) * self.sd_perturbation_0
            # Loop for 'many' epochs or until convergence
            for k in range(int(self.epochs *self.scale_factor)):
                print('Inner:', k, end = '|')
                self._optim_zeros()
                pert_inputs.requires_grad_()
                pre_loss = torch.zeros((self.num_adverse,1))
                for i,net in enumerate(self.net_list):
                    for j in range(self.num_adverse):
                        outputs = net.train()(pert_inputs[j,...])
                        # print('outputs shape:', outputs.shape,
                        #   'pert_inputs[j,...] shape:', pert_inputs[j,...].shape  )
                        pre_loss[j,:] +=  ( self.net_weights[i]*( self.criterion(outputs, targets) ) - 
                                        self.penalty_coef * self.adv_penalty(inputs, pert_inputs[j,...]) )
            #   Calculate the gradient with respect to each entry and move in this direction
                p0 = torch.autograd.grad(pre_loss,pert_inputs,create_graph=True, grad_outputs=torch.ones_like(pre_loss))[0]    
                with torch.no_grad():    
                    pert_inputs += self.eta['adv'](k)*p0.detach()
                    #   Evolve weights
                    self.adv_weights *=torch.exp( self.kappa['adv'] * pre_loss.squeeze() )
                    self.adv_weights/= self.adv_weights.sum()
            print(self.adv_weights)
            self._optim_zeros()
            loss = torch.zeros(self.n_learners)
            for i,net in enumerate(self.net_list):
                for j in range(self.num_adverse):
                    # print('i,j:',i,j)
                    outputs = net.train()(pert_inputs[j,...])
                    loss[i] +=  self.adv_weights[j] * (  self.criterion(outputs, targets) -  self.penalty_coef * self.adv_penalty(inputs, pert_inputs[j,...] ) )
            
            loss.sum().backward()
            self._optim_step()


            #self._optim_step()
            # Evolve weights
            self.net_weights *=torch.exp( -self.kappa['param'] * loss )
            self.net_weights/= self.net_weights.sum()


            # ESCRIBIR INFORMACION A MOSTRAR, IDEALMENTE CALCULAR GRADIENTES(?)




            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # outcome = predicted.data == targets
            # num_correct += outcome.sum().item()
            # progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | reg_term_1: %.3f '% \
            #  (train_loss/(batch_idx+1), 100.*num_correct/total, num_correct, total, reg_term_1/(batch_idx+1)  ))
            
        # self.train_loss.append(train_loss/(batch_idx+1))
        # self.train_acc.append(100.*num_correct/total)
        # self.train_reg.append(reg_term_1/(batch_idx+1))
                
    def test(self, epoch, num_pgd_steps=20):  
        '''
        Testing the model 
        '''
        test_loss, adv_acc, total, reg, clean_acc, grad_sum = 0, 0, 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.testloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net.eval()(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(targets).sum().item()
            total += targets.size(0)

            inputs_pert = inputs + 0.
            eps = 5./255.*8
            r = pgd(inputs, self.net.eval(), epsilon=[eps], targets=targets, step_size=0.04,
                    num_steps=num_pgd_steps, epsil=eps)

            inputs_pert = inputs_pert + eps * torch.Tensor(r).to(self.device)
            outputs = self.net(inputs_pert)
            probs, predicted = outputs.max(1)
            adv_acc += predicted.eq(targets).sum().item()
          

        print(f'epoch = {epoch}, adv_acc = {100.*adv_acc/total}, clean_acc = {100.*clean_acc/total}')

        self.test_loss.append(test_loss/(batch_idx+1))
        self.test_acc_adv.append(100.*adv_acc/total)
        self.test_acc_clean.append(100.*clean_acc/total)
        if self.test_acc_adv[-1] > self.test_acc_adv_best:
            self.test_acc_adv_best = self.test_acc_adv[-1]
            print(f'Saving the best model to {self.path}')
            self.save_model(self.path)
            
        return test_loss/(batch_idx+1), 100.*adv_acc/total, 100.*clean_acc/total

    


    def save_model(self, path):
        '''
        Saving the model
        ================================================
        Arguments:

        path: string
            path to save the model
        '''
        
        print('Saving...')

        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    


    def import_model(self, path):
        '''
        Importing the pre-trained model
        '''
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
           
            
    def plot_results(self):
        """
        Plotting the results
        """
        plt.figure(figsize=(15,12))
        plt.suptitle('Results',fontsize = 18,y = 0.96)
        plt.subplot(3,3,1)
        plt.plot(self.train_acc, Linewidth=2, c = 'C0')
        plt.plot(self.test_acc_clean, Linewidth=2, c = 'C1')
        plt.plot(self.test_acc_adv, Linewidth=2, c = 'C2')
        plt.legend(['train_clean', 'test_clean', 'test_adv'], fontsize = 14)
        plt.title('Accuracy', fontsize = 14)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14) 
        plt.grid()  
        plt.subplot(3,3,2)
        plt.plot(self.train_reg, Linewidth=2, c = 'C0')
        #plt.plot(self.test_curv, Linewidth=2, c = 'C1')
        #plt.legend(['train_curv', 'test_curv'], fontsize = 14)
        plt.legend(['train_reg'], fontsize = 14)
        plt.title('Regularization (order 1, q=1)', fontsize = 14)
        plt.ylabel('Reg', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.subplot(3,3,3)
        plt.plot(self.train_loss, Linewidth=2, c = 'C0')
        plt.plot(self.test_loss, Linewidth=2, c = 'C1')
        plt.legend(['train', 'test'], fontsize = 14)
        plt.title('Loss', fontsize = 14)
        plt.ylabel('loss', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()
        

