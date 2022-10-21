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
from torch.optim.swa_utils import AveragedModel, SWALR



class WAD2scale():
    def __init__(self, net_list, 
                    avg_nets,
                    trainloader, testloader, 
                    device = 'cuda', 
                    num_adverse = 6,
                    num_adverse_avg = 24,
                    scale_factor = 10,
                    kappa= None, 
                    eta = None, 
                    criterion = None,
                    adv_penalty = None,
                    penalty_coef = 3,
                    sd_perturbation_0 = 1,
                    max_batches = 10,
                    path='./checkpoint'):
        '''
        WAD2scale: 
        ================================================
        Arguments:

        net_list: List of PyTorch nn network structures
        avg_nets: 
        trainloader: PyTorch Dataloader
        testloader: PyTorch Dataloader
        device: 'cpu' or 'cuda' if GPU available type  to move tensors
        num_adverse: number of adversarial samples
        scale_factor: ratio of adversarial sample epoch per model epoch
        kappa= dictionary with constants for kappa for both 'adv' and 'param'
        eta =  dictionary with functions for eta for both 'adv' and 'param'
        criterion = Criterion for the function to minimise
        adv_penalty = Criterion for the pnealty for adversaruals
        penalty_coef = Coefficient in fornt of penalty of adversarials
        sd_perturbation_0 = standard deviation for Gaussian perturbation of initial data
        max_batches: maximum number of batches
        path: string
            path to save the best model

        Note: the total objective is top find min max criterion(x_tilde, y; nu) - adv_penalty*(x,x_tilde)
        
        '''      

        if not torch.cuda.is_available() and device=='cuda':
            raise ValueError("cuda is not available")

        self.net_list = []
        for net in net_list:
            self.net_list.append(net.to(device))
        
        self.avg_nets= []
        for net in avg_nets:
            self.avg_nets.append(net.to(device))
        
        self.num_adverse_avg = num_adverse_avg
        self.max_batches = max_batches
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
        self._restart_adv()
        self.n_avg_models = len(avg_nets)
        self.avg_net_weights = (torch.ones( self.n_avg_models )/self.n_avg_models).to(self.device)


    def _restart_adv(self):
        self._new_adv = True
        self.adv_samples={ 'original': [], 'adverse':[], 'weights':[],'labels': []  }
        self.avg_adv_samples={ 'original': [], 'adverse':[], 'weights':[],'labels': []  }


    def _restart_weights(self):
        self.net_weights = (torch.ones(self.n_learners )/self.n_learners).to(self.device)
        self.adv_weights = (torch.ones(self.num_adverse)/self.num_adverse).to(self.device)


    def _save_adv_samples(self, path):
        torch.save((self.adv_samples,self.avg_adv_samples),path)


    def _load_adv_samples(self,path):
        self.adv_samples, self.avg_adv_samples = torch.load(path)


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

    def predict(self,x):
        outputs = []
        for i, model in enumerate(self.avg_nets):
            outputs.append( self.net_weights[i]* model(x))
        return sum(outputs)

            
    def _sample_no_rep(self, weights, n):
        '''
        samples n elements from a list of probabilities. Returns a tuple of 
        positions and weights. Requires n<len(weights), a dummy position is returnes
        together with possibly zero entries.
        =======================
        Arguments:

        '''

        if torch.is_tensor(weights):
            weights_ = weights.detach().numpy()
            n_weights = (weights_>0).sum()
                

        if n_weights >= n:
            pos_out = np.random.choice(n_weights, n, replace = False, p=weights_[weights_>0] )
            w_out = weights.detach().clone()[weights>0][pos_out]
            w_out/= w_out.sum()
        else:
            pos_out = np.zeros(n, dtype= np.int_ )
            pos_out[:n_weights] = np.arange(n_weights)
            w_out = torch.zeros(n)
            w_out[:n_weights] = weights[weights>0].detach().clone()
        return w_out, pos_out #torch.from_numpy(pos_out)


    def _update_avg_adverse (self, time):
        '''
        calculate the running time average of adversarial distributions
        '''
        # print('\n length of weights:',len(self.avg_adv_samples['weights']))
        # print(self.avg_adv_samples['weights'])
        # print(self.adv_samples['weights'])

        if self._new_adv:
            self.avg_adv_samples = copy.deepcopy(self.adv_samples)
        else:
            for i, advs in enumerate(self.avg_adv_samples['adverse']):
                
                weights_all = torch.cat( ( (time/(time+1))* self.avg_adv_samples['weights'][i], (1/(time+1))*self.adv_samples['weights'][i]),0)
                # weights_all = [w*(time/(time+1)) for w in self.avg_adv_samples['weights'][i]] + [w/time for w in self.adv_samples['weights'][i]]
                adverse_all = torch.cat((advs,self.adv_samples['adverse'][i]),0)
                self.avg_adv_samples['weights'][i], aux = self._sample_no_rep(weights_all,self.num_adverse_avg)
                self.avg_adv_samples['adverse'][i] = adverse_all[aux].detach().clone()


    def _update_avg_model (self, time):
        '''
        calculate the running time average of model distributions
        '''    
        weights_all = torch.cat( ( (time/(time+1))* self.avg_net_weights, (1/(time+1))*self.net_weights),0)
        model_all =  self.avg_nets + self.net_list
        
        self.avg_net_weights, aux = self._sample_no_rep(weights_all, self.n_avg_models)
        # print('\n aux',aux.shape, aux.type())
        # print('model_all', model_all.shape)
        self.avg_nets = [ copy.deepcopy(model_all[i]) for i in aux]
        
        # copy.deepcopy(model_all[aux])

    # def _avg_model_update(self):
    #     for i,net in enumerate(self.net_list):
    #         self.avg_nets[i].update_parameters(net)


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
            print('\nEpoch: %d' % epoch)
            start_time = time.time()
            print(self.adv_weights)
            print(self.net_weights)
            self._train(epoch)
            #Calculate 'epoch average of the model'
            # self.test(epoch)
            self._sched_step()
            self._update_avg_model(epoch)
            self.train_times.append(time.time()-start_time)
            self._new_adv = False





    def _train(self, epoch):
        '''
        Training the model 
        '''

        # train_loss, total = 0, 0
        # num_correct = 0
        # reg, reg_term_1, norm_grad_sum  = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            #   Load inputs and target and 
            #   create copies of inputs and target (extend one rank to tensor)
            if batch_idx > self.max_batches:
                break
            print(batch_idx, end='|')
            inputs_lst = torch.tile(inputs, (self.num_adverse,1,1 ,1,1))
            
            # targets_lst = torch.tile(targets, (self.num_adverse,1))
            inputs_lst, targets = inputs_lst.to(self.device), targets.to(self.device)

            if self._new_adv:
                # create perturbed inputs and set adversarial weights and values
                self.adv_weights = self.adv_weights = (torch.ones(self.num_adverse)/self.num_adverse).to(self.device)
                pert_inputs = inputs_lst + torch.randn_like(inputs_lst).to(self.device) * self.sd_perturbation_0
            else:
                #Load adversarial weights and values
                self.adv_weights = copy.deepcopy(self.adv_samples['weights'][batch_idx])
                pert_inputs =  copy.deepcopy(self.adv_samples['adverse'][batch_idx])
    

               
            # Loop for 'many' epochs or until convergence
            for k in range(self.scale_factor):
                # print('Inner:', k, end = '|')
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
                    self.adv_weights *=torch.exp( self.kappa['adv'] * pre_loss.squeeze() ).to(self.device)
                    self.adv_weights/= self.adv_weights.sum()
            
            # Keep last version of adversarial images in memory
            if self._new_adv:
                self.adv_samples['original'].append(copy.deepcopy(inputs))
                self.adv_samples['labels'].append(copy.deepcopy(targets))
                self.adv_samples['adverse'].append(copy.deepcopy(pert_inputs.detach()))
                self.adv_samples['weights'].append(self.adv_weights)
            else:
                self.adv_samples['adverse'][batch_idx] = copy.deepcopy(pert_inputs)
                self.adv_samples['weights'][batch_idx] = self.adv_weights
            
            self._update_avg_adverse(epoch)


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
            outputs = self.predict(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(targets).sum().item()
            total += targets.size(0)

            inputs_pert = inputs + 0.
            eps = 5./255.*8
            i_rand = np.random.randint(0, self.n_learners)
            r = pgd(inputs, self.net_list[i_rand].eval(), epsilon=[eps], targets=targets, step_size=0.04,
                    num_steps=num_pgd_steps, epsil=eps)

            inputs_pert = inputs_pert + eps * torch.Tensor(r).to(self.device)
            outputs = self.predict(inputs_pert)
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
        Saving models and adversarial samples
        ================================================
        Arguments:

        path: string
            path to save the model
        '''
        
        print('Saving...')
        
        state = {}
        for i, net in enumerate(self.net_list):
            state['net_state_'+str(i)] = net.state_dict()
            state['optim_state_'+str(i)] = self.optim_list[i].state_dict()
            state['avg_net_state_'+str(i)] = self.avg_nets[i].state_dict()
        
        state['adv_samples'] = self.adv_samples
        state['avg_adv_samples'] = self.avg_adv_samples


        torch.save(state, path)
        print('done.')
    


    def import_model(self, path):
        '''
        Importing the pre-trained model
        ==============================
        Arguments:

        path: string
            path to where model is saved
        '''
        
        print('Loading...')
        checkpoint = torch.load(path)

        for i, net in enumerate(self.net_list):
            net.load_state_dict(checkpoint['net_state_'+str(i)])
            self.optim_list[i].load_state_dict(checkpoint['optim_state_'+str(i)])
            self.avg_nets[i].load_state_dict(checkpoint['avg_net_state_'+str(i)])
        
        self.adv_samples = checkpoint['adv_samples']
        self.avg_adv_samples = checkpoint['avg_adv_samples'] 

        print('done.')  
            
           
            
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
        

