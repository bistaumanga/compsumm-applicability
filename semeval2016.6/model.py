import torch
import torch.nn.functional as F

from commons.utils import get_logger
logger = get_logger("LOSS")

def weights_init(m):
    torch.manual_seed(19)
    classname = m.__class__.__name__
    # m.weight.data.fill_(0.0)
    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
    # torch.nn.init.xavier_uniform(m.weight)
    # m.weight.data.uniform_(-0.1, 0.1)
    m.bias.data.fill_(0.01)

class LinearModel1(torch.nn.Module):
    """ 
        simple linear classifier
    """
    def __init__(self, input_dims, output_dims = 3, **kwargs):
        super(LinearModel1, self).__init__()
        self._linear = torch.nn.Linear(input_dims, output_dims, bias = True)
        self._linear.apply(weights_init)
        # self._relu = torch.nn.ReLU()
        # self._dropout = torch.nn.Dropout(0.2)
        # self._activation = torch.nn.Softmax()
        self.name = "lin1"

    def print_wts(self):
        print(self._linear.weight.data)

    def forward(self, X):
        return self._linear(X)

class MLP2(torch.nn.Module):
    """ 
        simple linear classifier
    """
    def __init__(self, input_dims, hidden_dims, output_dims = 3, **kwargs):
        super(MLP2, self).__init__()
        self._linear1 = torch.nn.Linear(input_dims, hidden_dims, bias = True)
        self._linear1.apply(weights_init)
        self._relu = torch.nn.ReLU()
        self._dropout = torch.nn.Dropout(0.5)
        self._linear2 = torch.nn.Linear(hidden_dims, output_dims, bias = True)
        self._linear2.apply(weights_init)
        self._activation = torch.nn.Softmax()
        self.name = "mlp2"

    def forward(self, X):
        layer_1_op = self._relu(self._linear1(X))
        layer_1_op = self._dropout(layer_1_op)
        return self._linear2(layer_1_op)

# https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=5, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def __repr__(self):
        return "state:{:.6g}, #bad_epochs:{}".format(0.0 if self.best is None else self.best.item(), self.num_bad_epochs)

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)