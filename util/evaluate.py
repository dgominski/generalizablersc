import numpy as np
import warnings


def compute_anmrr(ranks, gnd):
    anmrr = 0.
    nq = len(gnd)  # number of queries
    nempty = 0

    # Browse first time to get nb positives per query
    qgnd = []
    ng = []
    for i in np.arange(nq):
        positives = np.atleast_1d(np.array(gnd[i]['positive'])).astype(int)
        qgnd.append(positives)
        ng.append(len(positives))

    gtm = np.max(ng)

    for i in np.arange(nq):
        k = min(4*ng[i], 2*gtm)

        pos = np.arange(ranks.shape[1])[np.in1d(ranks[i, :], qgnd[i])]

        qranks = pos
        qranks[qranks>k] = k+1

        avr = np.sum(qranks)/ng[i]

        mrr = avr - 0.5 - ng[i]/2

        nmrr = mrr/(k+0.5-0.5*ng[i])

        anmrr += nmrr

    anmrr /= nq

    return anmrr


def compute_recall(ranks, gnd, k):
    """
    Computes average recall @k:
    average number of correctly retrieved images over the whole set of positives in the first k results

    Arguments
    ---------
    ranks : ranked list of images in the database
    gnd  : ground truth

    Returns
    -------
    ar@k    : average recall @k
    """
    rk = 0

    nq = len(gnd)  # number of queries
    nempty = 0

    ranks = ranks.T
    for i in np.arange(nq):
        qgnd = np.atleast_1d(np.array(gnd[i]['positive'])).astype(int)

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['ignore'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and ignore images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        ignore = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        if len(np.intersect1d(pos, ignore)) != 0:
            raise ValueError("There is a common element between positive and ignore indexes for image # {}".format(i))

        t = 0
        ij = 0
        if len(ignore):
            # decrease positions of positives based on the number of
            # ignore images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(ignore) and pos[ip] > ignore[ij]):
                    t += 1
                    ij += 1
                pos[ip] = pos[ip] - t
                ip += 1

        rk += np.sum(pos<k) / len(qgnd)

    rk /= nq
    return rk


def compute_recall_loc(ranks, gnd, k):
    """
    Computes average recall (localization definition):
    percentage of query images with at least one relevant database image
    amongst the topk retrieved ones.

    Arguments
    ---------
    ranks : ranked list of images in the database
    gnd  : ground truth

    Returns
    -------
    ar@k    : average recall @k
    """
    rk = 0

    nq = len(gnd)  # number of queries
    nempty = 0

    ranks = ranks.T
    for i in np.arange(nq):
        qgnd = np.atleast_1d(np.array(gnd[i]['positive'])).astype(int)

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['ignore'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and ignore images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        ignore = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        if len(np.intersect1d(pos, ignore)) != 0:
            raise ValueError("There is a common element between positive and ignore indexes for image # {}".format(i))

        t = 0
        ij = 0
        if len(ignore):
            # decrease positions of positives based on the number of
            # ignore images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(ignore) and pos[ip] > ignore[ij]):
                    t += 1
                    ij += 1
                pos[ip] = pos[ip] - t
                ip += 1

        rk += np.sum(pos<k)>=1

    rk /= nq
    return rk


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map_standard(ranks, gnd, kappas=[1, 5, 10]):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The ignore results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    ranks = ranks.T
    for i in np.arange(nq):
        qgnd = np.atleast_1d(np.array(gnd[i]['positive'])).astype(int)

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['ignore'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and ignore images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        ignore = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        if len(np.intersect1d(pos, ignore)) != 0:
            raise ValueError("There is a common element between positive and ignore indexes for image # {}".format(i))

        k = 0
        ij = 0
        if len(ignore):
            # decrease positions of positives based on the number of
            # ignore images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(ignore) and pos[ip] > ignore[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        if pos.size != 0:
            pos += 1  # get it to 1-based
            for j in np.arange(len(kappas)):
                kq = min(max(pos), kappas[j])
                prs[i, j] = (pos <= kq).sum() / kq
            pr = pr + prs[i, :]

    if nq - nempty == 0:
        warnings.warn("mAP score can't be evaluated due to empty ground truth")
        return 0.0, aps, pr, prs

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map(dataset_name, ranks, gnd, kappas=[1, 5, 10], verbose=True):
    map, aps, pr, prs = compute_map_standard(ranks, gnd, kappas)

    if verbose:
        print('>> {}: mAP {:.2f}'.format(dataset_name, np.around(map * 100, decimals=2)))

    return map, aps, pr, prs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name="unkown", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(batch_size*k).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Batch_mAP(object):
    """
    Computes the mAP on a batch of images. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    n_samples is the number of samples per class
    """

    def __init__(self, n_classes, n_samples):
        self.n_classes = n_classes
        self.n_samples = n_samples

    def compute(self, feats):
        vecs = feats.cpu().numpy()
        qvecs = vecs

        scores = np.dot(vecs, qvecs.T).T
        ranks = np.argsort(-scores, axis=1)

        map = 0.
        nq = ranks.shape[0]  # number of queries
        aps = np.zeros(nq)

        ranks = ranks.T
        for i in np.arange(nq):
            qgnd = np.arange((i//self.n_samples)*self.n_samples, ((i//self.n_samples)*self.n_samples)+self.n_samples).astype(int)
            qgnd = np.delete(qgnd, i%self.n_samples)

            # sorted positions of positive images(0 based)
            pos = np.arange(ranks.shape[1])[np.in1d(ranks[:, i], qgnd)]

            # decrease positions of positives based on the number of
            # ignore images appearing before them
            ip = 0
            ij = 0
            k = 0
            while ip < len(pos):
                while ij < 1 and pos[ip] > 0:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

            # compute ap
            ap = compute_ap(pos, len(qgnd))
            map = map + ap
            aps[i] = ap


        map = map / nq

        return map, aps

