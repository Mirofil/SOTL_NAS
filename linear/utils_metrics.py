import torch
import itertools 
def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class ValidAccEvaluator:
  def __init__(self, valid_loader, valid_loader_iter=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    self.valid_loader = valid_loader
    self.valid_loader_iter=valid_loader_iter
    self.device = device
    super().__init__()

  def evaluate(self, network, criterion):
    network.eval()
    with torch.no_grad():
      try:
        inputs, targets = next(self.valid_loader_iter)
      except:
        self.valid_loader_iter = iter(self.valid_loader)
        inputs, targets = next(self.valid_loader_iter)
      logits = network(inputs.to(self.device))
      loss = criterion(logits, targets.to(self.device))
      if logits.shape[1] != 1:
        val_acc_top1, val_acc_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
        val_acc_top1, val_acc_top5 = val_acc_top1.item(), val_acc_top5.item()
      else:
        val_acc_top1, val_acc_top5 = None, None
    network.train()
    return val_acc_top1, val_acc_top5, loss.item()

class SumOfWhatever:
  def __init__(self, measurements=None, e = 1, epoch_steps=None, mode="sum"):
    if measurements is None:
      self.measurements = []
      self.measurements_flat = []
    else:
      self.measurements = measurements
      self.measurements_flat = list(itertools.chain.from_iterable(measurements))
    self.epoch_steps = epoch_steps
    self.e =e
    self.mode = mode

  def update(self, epoch, val):

    while epoch >= len(self.measurements):
      self.measurements.append([])
    self.measurements[epoch].append(val)
    self.measurements_flat.append(val)

  def get_time_series(self, e=None, mode=None, window_size = None, chunked=False):
    if mode is None:
      mode = self.mode

    params = self.guess(e=e, mode=mode, epoch_steps=None)
    return_fun, e, epoch_steps = params["return_fun"], params["e"], params["epoch_steps"]
    window_size = e*epoch_steps if window_size is None else window_size
    ts = []
    for step_idx in range(len(self.measurements_flat)):
      
      at_the_time = self.measurements_flat[max(step_idx-window_size+1,0):step_idx+1]
      ts.append(return_fun(at_the_time))
    if chunked is False:
      return ts
    else:
      return list(chunks(ts, epoch_steps))

    
  def guess(self, epoch_steps, e, mode):
    if mode == "sum":
      return_fun = sum
    elif mode == "last":
      return_fun = lambda x: x[-1]
    elif mode == "first":
      return_fun = lambda x: x[0]
    elif mode == "fd":
      return_fun = lambda x: x[-1] - x[-2] if len(x) >= 2 else 0
    elif mode == "R":
      return_fun = lambda x: -(x[-1] - x[-2]) + x[0] if len(x) >= 2 else x[0]


    if self.epoch_steps is None:
      epoch_steps = len(self.measurements[0])
    else:
      epoch_steps = self.epoch_steps

    if e is None:
      e = self.e

    return {"e":e, "epoch_steps":epoch_steps, "return_fun":return_fun}