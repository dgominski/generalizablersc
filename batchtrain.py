import os
from torch.utils.tensorboard import SummaryWriter
from options.options import Options
from data import create_dataset, multi_dataset, find_dataset_using_name
from models import create_model
import tqdm
import csv


if __name__ == '__main__':
    opt = Options().parse()  # get training options
 
    testdatasets = create_dataset(opt, mode='test')

    total_iters = 0  # the total number of training iterations
    val_iters = 0

    f = open("results.csv", "a")
    writer = csv.writer(f)
    header = ['targetdataset', '1shot-mahalanobis', '5shot-mahalanobis', '1shot-mahalanobis+alphaQE', '5shot-mahalanobis+alphaQE', '1shot-reducedproto', '5shot-reducedproto', '1shot-reducedproto+KRR', '5shot-reducedproto+KRR']
    writer.writerow(header)
    print(header)
    f.close()
    
    targetdatasets = ['siriwhu', 'ucm', 'whurs19', 'aid', 'patternnet', 'resisc45', 'rsicb', 'rsscn7'] 
    for targetdataset in targetdatasets:
        trainingdatasets = list(targetdatasets)
        trainingdatasets.remove(targetdataset)
        print("############### Target: {} Training on: {}".format(targetdataset, trainingdatasets))
        dataset = multi_dataset.MultiDataset(opt, trainingdatasets)
        
        testdataset = find_dataset_using_name(targetdataset, mode='test')(opt)
        for idmodel in range(5):
            opt.name = "{}_run_{}".format(targetdataset, str(idmodel))
            writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "tbx"))
            model = create_model(opt, mode='train')  # create a model given opt.model and other options
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            for epoch in range(opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

                dataloader = dataset.prepare_epoch()
                dataset_size = len(dataloader)  # get the number of images in the dataset.
                print('>> Epoch {}/{}: training...'.format(epoch, opt.niter+opt.niter_decay))

                for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):  # inner loop within one epoch
                    model.train()
                    total_iters += opt.batch_size
                    model.set_input(data)  # unpack data from dataset and apply preprocessing
                    model.forward()
                    model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                print('End of epoch {} / {}'.format(epoch, opt.niter+opt.niter_decay))

                model.update_learning_rate()  # update learning rates at the end of every epoch.
            
            f = open("results.csv", "a")
            writer = csv.writer(f)
            results = model.test(testdataset)
            for j in range(5):
                resultline = [targetdataset, results[1]['mahalanobis'][j], results[5]['mahalanobis'][j], results[1]['mahalanobis_aqe'][j], results[5]['mahalanobis_aqe'][j], results[1]['reducedproto'][j], results[5]['reducedproto'][j], results[1]['reducedproto_krr'][j], results[5]['reducedproto_krr'][j]]
                writer.writerow(resultline)
            f.close()    
