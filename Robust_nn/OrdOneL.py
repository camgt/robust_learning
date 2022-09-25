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
    
class OrdOneL():
    def __init__(self, net, trainloader, testloader, device='cuda', delta = 0.2, q=1, r=2,
                 path='./checkpoint'):
        '''
        ORDONEL: 
        ================================================
        Arguments:

        net: PyTorch nn
            network structure
        trainloader: PyTorch Dataloader
        testloader: PyTorch Dataloader
        device: 'cpu' or 'cuda' if GPU available
            type of decide to move tensors
        delta: float
            radius of Wasserstein ball
        q: float larger than one
        	power of norm
        path: string
            path to save the best model
        '''

        def dual(r):
            if np.isinf(r):
                rstar=1.
            else:
                rstar = r/(r-1)
            return rstar


        if not torch.cuda.is_available() and device=='cuda':
            raise ValueError("cuda is not available")

        self.net = net.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.q = q
        self.r = r
        self.rstar = dual(r)        
        self.delta = delta
        self.trainloader, self.testloader = trainloader, testloader
        self.path = path
        self.test_acc_adv_best = 0
        self.train_loss, self.train_acc, self.train_reg = [], [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean, self.test_reg = [], [], [], []    
        self.train_times=[]

    def set_optimizer(self, optim_alg='Adam', args={'lr':1e-4}, scheduler=None, args_scheduler={}):
        '''
        Setting the optimizer of the network
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
        self.optimizer = getattr(optim, optim_alg)(self.net.parameters(), **args)
        if not scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10**6, gamma=1)
        else:
            self.scheduler = getattr(optim.lr_scheduler, scheduler)(self.optimizer, **args_scheduler)
        
            
    def train(self, epochs = 15, delta = None):
        '''
        Training the network
        ================================================
        Arguemnets:

        h : list with length less than the number of epochs
            Different h for different epochs of training, 
            can have a single number or a list of floats for each epoch
        epochs : int
            Number of epochs
        '''
        if delta == None:
            delta = [self.delta]        
        if len(delta)>epochs:
           raise ValueError('Length of delta should be less than number of epochs')
        if len(delta)==1:
           delta = epochs * [delta[0]]
        else:
           d_all = epochs * [1.0]
           d_all[:len(delta)] = list(delta[:])
           d_all[len(delta):] = (epochs - len(delta)) * [delta[-1]]
           delta = d_all


        for epoch in range(epochs):
            start_time = time.time()
            self._train(epoch, delta[epoch])
            self.test(epoch)
            self.scheduler.step()
            self.train_times.append(time.time()-start_time)
        
    def _train(self, epoch, delta):
        '''
        Training the model 
        '''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        reg, reg_term_1, norm_grad_sum  = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)            
            inputs.requires_grad_()
            outputs = self.net.train()(inputs)                        
            pre_loss = self.criterion(outputs, targets) 
            p0 = -1.0* inputs.shape[0]*torch.autograd.grad(pre_loss,inputs,create_graph=True, grad_outputs=torch.ones_like(pre_loss))[0]                                    
            p0norm = torch.linalg.norm(p0.reshape(p0.shape[0],-1), self.rstar, dim=-1)
            # p0norm = ((p0**2).sum(dim=(1,2,3)))**0.5
            # if epoch<=1 and batch_idx == 0:
            #     print(pre_loss.shape,inputs.shape, targets.shape, p0.shape,p0norm.shape )
            reg_loss = delta* p0norm.mean()
            loss = pre_loss + reg_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            reg_term_1+= reg_loss.item() 
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            outcome = predicted.data == targets
            num_correct += outcome.sum().item()
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | reg_term_1: %.3f '% \
             (train_loss/(batch_idx+1), 100.*num_correct/total, num_correct, total, reg_term_1/(batch_idx+1)  ))
            
        self.train_loss.append(train_loss/(batch_idx+1))
        self.train_acc.append(100.*num_correct/total)
        self.train_reg.append(reg_term_1/(batch_idx+1))
                
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
        



class OrdTwoL(OrdOneL):
    def __init__(self, net, trainloader, testloader, device='cuda', delta = 0.2, q=1,r=2, o1 = True, o2 =  True, mod_o2 = False,
                 path='./checkpoint'):
        super().__init__(net, trainloader, testloader, device=device, delta = delta, q=q,r=r,
                 path='./checkpoint')
        self.o1=o1
        self.o2=o2
        self.mod_o2=mod_o2
        

    def _train(self, epoch, delta):
        '''
        Training the model 
        '''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        reg, reg_term_1, norm_grad_sum  = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)            
            inputs.requires_grad_()
            outputs = self.net.train()(inputs)                        
            pre_loss = self.criterion(outputs, targets) 
            p0 = -1.0* inputs.shape[0]*torch.autograd.grad(pre_loss,inputs,create_graph=True, grad_outputs=torch.ones_like(pre_loss))[0]                                    
            auxp = p0.reshape(p0.shape[0],-1)
            p0norm = torch.linalg.norm(auxp, self.rstar, dim=-1)            
                        
            if self.r==1:
                posmax = torch.argmax(auxp, dim=1)
                auxp2 = torch.zeros_like(auxp)
                auxp2[:,posmax]=1.
            elif np.isinf(self.r):
                auxp2 = torch.sign(auxp)
            else:                
                auxp2 = (auxp**(self.rstar-1) + 1e-7) / (p0norm[:,None]**(self.rstar-1)+1e-7)

            reg_loss = 0 

            if self.mod_o2:
                auxp2 = auxp2.detach()
                alpha = torch.autograd.grad( (auxp*auxp2).sum(dim=-1)  ,inputs,create_graph=True, grad_outputs= torch.ones_like(p0norm) )[0]            
            elif self.o1 or self.o2:
                alpha = torch.autograd.grad(p0norm,inputs,create_graph=True, grad_outputs= torch.ones_like(p0norm) )[0]            
            
            if self.o1:
                reg_loss += delta* p0norm.mean()            
            if self.mod_o2:
                    reg_loss +=delta**2/2* (auxp2*alpha.reshape(p0.shape[0],-1)).mean()                                        
            
            loss = pre_loss + reg_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.o1 or self.o2:
                reg_term_1+= reg_loss.item() 
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            outcome = predicted.data == targets
            num_correct += outcome.sum().item()
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | reg_term_1: %.3f '% \
             (train_loss/(batch_idx+1), 100.*num_correct/total, num_correct, total, reg_term_1/(batch_idx+1)  ))
            
        self.train_loss.append(train_loss/(batch_idx+1))
        self.train_acc.append(100.*num_correct/total)
        self.train_reg.append(reg_term_1/(batch_idx+1)) 
    
    

    


