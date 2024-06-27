# %% [markdown]
# # Wasserstein Ascend Descend
# ### (C. and N.) Garcia Trillos


# %%
from Robust_nn.WAD import WAD2scale
from utils.utils import read_vision_dataset
from utils.convnet import ConvNet
from utils.convnet_silu import ConvNetSiLU
import torch
import torch.nn as nn



def generate_markdown_table(results, file_path='results_WAD.md'):
        with open(file_path, 'w') as f:
            # Write the header row
            f.write('| Metric | Value |\n')
            f.write('| ------ | ----- |\n')
            
            # Write the data rows
            for metric, value in results.items():
                f.write(f'| {metric} | {value} |\n')
        
        print(f'Markdown file {file_path} generated successfully.')



def test (model, avg_type, results):
     
    # Train the model
    model.set_optimizer()
    model.train(epochs=training_epochs)

    # Saving the model
    model.save_model('Test_' + avg_type)

    # Test the model accuracy
    avg_loss, PGD_acc, clean_acc = adv_net.test_pgd(PGD_steps)

    # Test the relative change at final value
    val_at_nash = adv_net.test_base()
    val_better_adv = adv_net.test_improve_adversaries(additional_training_epochs)
    val_better_model = adv_net.test_improve_model(additional_training_epochs)

    rel_adv = abs(val_better_adv/val_at_nash -1)
    rel_model = abs(val_better_model/val_at_nash -1)

    # Add results to output structure
    results['clean_accuracy_'+avg_type] = clean_acc
    results['PGD_accuracy_'+avg_type] = PGD_acc
    results['rel_change_adv_'+avg_type] = rel_adv
    results['rel_change_model_'+avg_type] = rel_model



if __name__ == '__main__':  
    '''
        Testing of the models as in the paper. Outputs are written in a markdown file: 'results.md'.
    '''
    
    
    
    # %%
    # Define parameters of test

    N = 4               # Number of particles, in this case neural networks
    M = 2               # Number of adversaries, in this case distorted images
    eta_t = {'param': lambda t: 0.1/(t+1), 'adv': lambda t : 0.1/(t+1)}
    kappa = { 'param': 0.25, 'adv': 0.25  }
    ca = 10             # Cost coefficient (penalty) 
    batch_size = 64      
    training_epochs = 4 
    dataset_name = 'MNIST'
    PGD_steps = 20
    additional_training_epochs = 5


    max_batches = 937
    results = {}
    
    # ============================
    # Time average on weights model
    # ================================

    print('== Tests for time average on weight == ')
    # Define networks and model
    net_lst = [ConvNet() for i in range(N)]
    avg_nets =[ConvNet() for i in range(N*4)]
                        
    adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
                        dataset_name= dataset_name,batch_size = batch_size, 
                        device = None , criterion= nn.CrossEntropyLoss(), 
                        scale_factor=5, num_adverse= M,
                        penalty_coef = ca,
                        kappa = kappa,
                        max_batches= max_batches)

    # Perform tests

    test(adv_net, 'timeAvg', results )

    # ============================
    # Resampling average on weights model
    # ================================

    print('== Tests for resampling average == ')
    
    # Define networks and model
    net_lst = [ConvNet() for i in range(N)]
    avg_nets =[ConvNet() for i in range(N*4)]
                        
    adv_net = WAD2scale(net_list = net_lst, avg_nets = avg_nets, 
                        dataset_name= dataset_name,batch_size = batch_size, 
                        device = None , criterion= nn.CrossEntropyLoss(), 
                        scale_factor=5, num_adverse= M,
                        penalty_coef = ca,
                        kappa = kappa,
                        max_batches= max_batches, average_weights_only = False)

    # Perform tests
    test(adv_net, 'resample', results )


    # ========
    # Save results
    # =======

    print('== Saving results == ')
    generate_markdown_table(results)
    print('== done == ')
