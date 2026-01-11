import torch
import os

class config():
    # BayOpt
    min_lamda = 0.5
    max_lamda = 1
    min_mut_rate = 0.5
    max_mut_rate = 1
    min_num_model = 1
    max_num_model = 5
    step_num_model = 1
    max_evals = 10
    save_path = ''

    # Bayobjective
    it_num = 10
    pop_num = 15
    CS = 2 * pop_num

    num_classes = 2
    batchsize = 16
    epoch_num = 100
    LR = 0.001
    best_loss = 100000
    early_stop = 0
    max_early_stop = 10
    modelsavefile = ''
    lossfile = ''
    datafile = ''

    T = 10

    def device_detection(self):
        print("PYTORCH's version is ", torch.__version__)
        #os.environ['CUDA_VISIBLE_DEVICES']='1'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: ', device)
        print('gpu数量:',torch.cuda.device_count())
        torch.set_num_threads(5)
        return device