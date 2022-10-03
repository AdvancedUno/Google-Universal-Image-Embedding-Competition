import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, roc_auc_score


class Metric:
    def __init__(self):
        pass
    
    def __call__(self, outputs, targets):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    
    def value(self):
        raise NotImplementedError()
    
    def name(self):
        raise NotImplementedError()


class Accuracy(Metric):
    def __init__(self, topK):
        super().__init__()
        self.topK = topK
        self.reset()

    def __call__(self, outputs, targets):
        _, pred = outputs.topk(self.topK, 1, True, True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        self.correct_k = correct.view(-1).float().sum(0)
        self.total = targets.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0 

    def value(self):
        return float(self.correct_k) / self.total

    def  name(self):
        return 'accuracy'


class F1Score(Metric):
    def __init__(self, thresh=0.5, normalizate=True, task_type='multiclass', average='micro', search_thresh=False):
        super(F1Score).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'


class AUC(Metric):
    def __init__(self, task_type='binary', average='binary'):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self, logits, target):
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return 'auc'


class Pearson(object):
    def __init__(self):
        super(Pearson, self).__init__()
        self.reset()

    def __call__(self, logits, target):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        self.score = pearsonr(target.squeeze(), logits.squeeze())[0] * 100

    def reset(self):
        self.score = 0

    def value(self):
        return round(float(self.score), 2)

    def name(self):
        return 'pearson'


class Spearman(object):
    def __init__(self):
        super(Spearman,self).__init__()
        self.reset()

    def __call__(self, logits, target):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        self.score = spearmanr(target.squeeze(), logits.squeeze())[0] * 100

    def reset(self):
        self.score = 0

    def value(self):
        return round(float(self.score), 2)

    def name(self):
        return 'spearman'
