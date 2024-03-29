from options.options import Options
from data import create_dataset
from models import create_model


if __name__ == '__main__':
    opt = Options().parse()  # get training options

    model = create_model(opt, mode='test')  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    testdataset = create_dataset(opt, mode='test')  # create a dataset given opt.train_dataset_mode and other options
    dataset_size = len(testdataset)  # get the number of images in the dataset.
    print("Evaluating on dataset {} size {}".format(testdataset.name, dataset_size))

    print(model.test(testdataset))
