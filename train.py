import os
from torch.utils.tensorboard import SummaryWriter
from options.options import Options
from data.multi_dataset import MultiDataset
from models import create_model
import tqdm
from data import create_dataset


def val(dataset, model, writer, epoch):
    print("Running validation")
    model.eval()
    results = model.test(dataset, mode="val")
    writer.add_scalars("val/{}".format(dataset.name), results, global_step=epoch)
    model.train()
    return


if __name__ == '__main__':
    opt = Options().parse()  # get training options

    alldatasets = ['aid', 'siriwhu', 'whurs19', 'patternnet', 'resisc45', 'rsicb', 'rsscn7', 'ucm']
    alldatasets.remove(opt.test_dataset)
    dataset = MultiDataset(opt, alldatasets)

    testdatasets = create_dataset(opt, mode='test')

    model = create_model(opt, mode='train')  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "tbx")) if not opt.debug else SummaryWriter('TMP')

    total_iters = 0  # the total number of training iterations
    
    for d in dataset.val_datasets:
        val(d, model, writer, epoch=0)

    model.eval()
    for testdataset in dataset.test_datasets:
        results = model.test(testdataset, mode="test")
        writer.add_scalars("test/{}".format(testdataset.name), results, 0)
    
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        print("Training...")
        model.train()
        dataloader = dataset.prepare_epoch()
        print('>> Epoch {}/{}: training...'.format(epoch, opt.niter+opt.niter_decay))

        for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):  # inner loop within one epoch
            model.train()
            total_iters += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
        
            if total_iters//opt.batch_size % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                writer.add_scalars("train/loss", losses, global_step=total_iters)
                scalars = model.get_current_scalars()
                writer.add_scalars("train/scalars", scalars, global_step=total_iters)
        
        print('End of epoch {} / {}'.format(epoch, opt.niter + opt.niter_decay))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.eval()
        for testdataset in dataset.test_datasets:
            results = model.test(testdataset, mode="test")
            writer.add_scalars("test/{}".format(testdataset.name), results, epoch)

        for d in dataset.val_datasets:
            val(d, model, writer, epoch)

