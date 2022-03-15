import numpy as np
import torch
from models.base_model import BaseModel
torch.backends.cudnn.benchmark = True
import models.networks as nets
from layers.pooling import MAC, SPoC, GeM, RMAC
from util.evaluate import accuracy, compute_map, Batch_mAP
from util.util import set_bn_eval
from layers.loss import ArcMarginProduct, SmoothAP
from models.networks import init_net
import tqdm
from collections import OrderedDict
from util.diffusion import alpha_query_expansion


POOLING = {
    'mac'   : MAC,
    'spoc'  : SPoC,
    'gem'   : GeM,
    'rmac'  : RMAC,
}

OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
}


class deepdescModel(BaseModel, torch.nn.Module):
    def __init__(self, opt, mode, num_classes):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.isTrain = mode == 'train'
        self.loss_names = ['ranking']
        self.scalar_names = ['niter', 'batchmap']
        self.model_names = ['feature_extractor', 'whiten']

        self.n_samples_per_class = opt.k
        
        # Feature extraction network
        self.netfeature_extractor = nets.define_feature_extractor(opt.net, opt.pretrained)
        self.netfeature_extractor = init_net(self.netfeature_extractor, pretrained=opt.pretrained, gpu_ids=opt.gpu_ids)

        # Normalization layer
        self.norm = nets.L2N()

        # Pooling layer
        self.pool = POOLING[opt.pool]()
        self.pool = init_net(self.pool, pretrained=True, gpu_ids=opt.gpu_ids)

        self.netwhiten = torch.nn.Sequential(
                torch.nn.Linear(OUTPUT_DIM[opt.net], opt.dim, bias=True),
                torch.nn.BatchNorm1d(opt.dim)
            ).to(self.device)
        self.netwhiten = init_net(self.netwhiten, pretrained=False, gpu_ids=opt.gpu_ids)
        
        # Loss function
        self.ranking_criterion = SmoothAP(anneal=0.01, batch_size=int(opt.batch_size*self.n_samples_per_class), num_id=int(self.n_samples_per_class), feat_dims=self.opt.dim, device=self.device)

        self.evaluator = Batch_mAP(n_classes=int(opt.batch_size), n_samples=int(self.n_samples_per_class))
        
        # Optimizer
        self.optimizers['feature_extractor'] = torch.optim.Adam(list(self.netfeature_extractor.parameters()) + list(self.netwhiten.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.niter = 0
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        self.input = input.get('input', torch.zeros(1, 1, 1, 1)).to(self.device)
        self.target = input.get('target', torch.Tensor([-1, 1])).to(self.device)
        self.sampleclass = input.get('class', torch.zeros(1)).to(self.device).squeeze().long()
        self.datasets = np.array(input.get('dataset'))
        self.paths = input.get('path')
        self.unique_datasets = np.unique(self.datasets)

    def freeze_batchnorm(self):
        print("freezing batchnorm")
        self.netfeature_extractor.apply(set_bn_eval)
        self.netwhiten.apply(set_bn_eval)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # x -> features
        self.deepfeats = self.netfeature_extractor(self.input)

        # features -> pool -> norm
        o = self.norm(self.pool(self.deepfeats)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.netwhiten is not None:
            o = self.norm(self.netwhiten(o))

        self.deepdesc = o
        return self.deepdesc

    def __call__(self, image_input):
        self.set_input({'input': image_input})
        return self.forward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.batchmap = 100*self.evaluator.compute(self.deepdesc.detach())[0]
        
        self.loss_ranking = self.ranking_criterion(self.deepdesc)

        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        self.deepdesc.retain_grad()
        self.loss_ranking.backward()

        for name in self.optimizers:
            self.optimizers[name].step()

        self.niter += self.opt.batch_size
        return {'train_loss': self.loss_ranking.data.cpu()}

    def batch_extract(self, loader):
        """Extract descriptors for a large batch of input images, no gradients"""
        self.eval()
        self.freeze_batchnorm()
        with torch.no_grad():
            # extract query vectors
            descs = []
            for i, input in tqdm.tqdm(enumerate(loader), total=len(loader)):
                self.set_input(input)
                descs.append(self.forward())
            descs = torch.cat(descs, dim=0)
        return descs

    def test(self, dataset, mode="test"):
        self.eval()
        with torch.no_grad():
            results = {}
            for nshot in [1,5]:
                mahalanobis = []
                mahalanobis_aqe = []
                for run in range(5):
                    print('>> RSC: Evaluating network on {} {} split...'.format(dataset.name, mode))
            
                    # get class prototypes
                    classes, protoinputs, loader, gnd = dataset.get_protos_and_loader(nshot)
                    classes = classes.astype(np.int)

                    # extract database and query vectors
                    print(">>> Extracting database descriptors")
                    qvecs = self.batch_extract(loader).cpu()
                    protos = []
                    for uclass in np.unique(classes):
                        self.set_input({'input': protoinputs[classes==uclass]})
                        self.forward()
                        protos.append(self.deepdesc)
                    protos = torch.cat(protos, dim=0)
                    
                    context_features, target_features = protos.cpu(), qvecs.cpu()
                    classes = torch.from_numpy(classes).cpu()
                    class_means, class_precision_matrices = self.build_class_reps_and_covariance_estimates(context_features, classes)
                    class_means = torch.stack(list(class_means.values())).squeeze(1)
                    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

                    # grabbing the number of classes and query examples for easier use later in the function
                    number_of_classes = class_means.size(0)
                    number_of_targets = target_features.size(0)

                    """
                    SCM: calculating the Mahalanobis distance between query examples and the class means
                    including the class precision estimates in the calculations, reshaping the distances
                    and multiplying by -1 to produce the sample logits
                    """
                    repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
                    repeated_class_means = class_means.repeat(number_of_targets, 1)
                    repeated_difference = (repeated_class_means - repeated_target)
                    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
                    first_half = torch.matmul(repeated_difference, class_precision_matrices)
                    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1

                    test_logits_sample = split_first_dim_linear(sample_logits, [1, target_features.shape[0]])
                    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
                    accuracy = torch.mean(torch.eq(torch.from_numpy(gnd.astype(int)), torch.argmax(averaged_predictions, dim=-1)).float())
                    mahalanobis.append(accuracy.cpu().numpy())

                    # with diffusion
                    n_protos = protos.shape[0]
                    all_vecs = torch.cat((protos.cpu(), qvecs), dim=0)
                    newqvecs = torch.from_numpy(alpha_query_expansion(all_vecs, alpha=3, n=10)[n_protos:])
                    context_features, target_features = protos.cpu(), newqvecs.cpu()
                    classes = classes.cpu()
                    class_means, class_precision_matrices = self.build_class_reps_and_covariance_estimates(context_features, classes)
                    class_means = torch.stack(list(class_means.values())).squeeze(1)
                    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

                    # grabbing the number of classes and query examples for easier use later in the function
                    number_of_classes = class_means.size(0)
                    number_of_targets = target_features.size(0)

                    """
                    SCM: calculating the Mahalanobis distance between query examples and the class means
                    including the class precision estimates in the calculations, reshaping the distances
                    and multiplying by -1 to produce the sample logits
                    """
                    repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
                    repeated_class_means = class_means.repeat(number_of_targets, 1)
                    repeated_difference = (repeated_class_means - repeated_target)
                    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
                    first_half = torch.matmul(repeated_difference.double(), class_precision_matrices.double())
                    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1

                    test_logits_sample = split_first_dim_linear(sample_logits, [1, target_features.shape[0]])
                    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
                    accuracy = torch.mean(torch.eq(torch.from_numpy(gnd.astype(int)), torch.argmax(averaged_predictions, dim=-1)).float())
                    mahalanobis_aqe.append(accuracy.cpu().numpy())

                results['mahalanobis_'+str(nshot)+'shot_mean'] = np.mean(mahalanobis)
                results['mahalanobis_'+str(nshot)+'shot_std'] = np.std(mahalanobis)
                results['mahalanobis_aqe'+str(nshot)+'shot_mean'] = np.mean(mahalanobis_aqe)
                results['mahalanobis_aqe'+str(nshot)+'shot_std'] = np.std(mahalanobis_aqe)
            
            return results

    def build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        SCM: calculating a task level covariance estimate using the provided function.
        """
        class_representations = OrderedDict()
        class_precision_matrices = OrderedDict()
        task_covariance_estimate = self.estimate_cov(context_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = torch.mean(class_features, dim=0, keepdim=True)
            # updating the class representations dictionary with the mean pooled representation
            class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                    + torch.eye(class_features.size(1), class_features.size(1)))

        return class_representations, class_precision_matrices

    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
        Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)
