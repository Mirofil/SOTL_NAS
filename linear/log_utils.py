import wandb
import os
import matplotlib.pyplot as plt
import torch

def visualize_ae(dset_test, model):
  n = 10  # How many digits we will display
  test_loader = torch.utils.data.DataLoader(dset_test, batch_size=n, shuffle=True)
  batch=next(iter(test_loader))
  plt.figure(figsize=(20, 4))
  model.to('cpu')
  for i in range(n):
      # Display original
      x, y = batch[i]
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction
      decoded_img = model(x)
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(decoded_img.reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()
def wandb_auth(fname: str = "nas_key.txt"):
    gdrive_path = "/content/drive/MyDrive/Colab Notebooks/wandb/nas_key.txt"

    if "WANDB_API_KEY" in os.environ:
        wandb_key = os.environ["WANDB_API_KEY"]
    elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
        # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
        print("Retrieving WANDB key from file")
        f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
        key = f.read().strip()
        os.environ["WANDB_API_KEY"] = key
    elif os.path.exists("/root/.wandb/"+fname):
        print("Retrieving WANDB key from file")
        f = open("/root/.wandb/"+fname, "r")
        key = f.read().strip()
        os.environ["WANDB_API_KEY"] = key

    elif os.path.exists(
        os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
    ):
        print("Retrieving WANDB key from file")
        f = open(
            os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
            "r",
        )
        key = f.read().strip()
        os.environ["WANDB_API_KEY"] = key

    elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
    wandb.login()


class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val   = 0.0
    self.avg   = 0.0
    self.sum   = 0.0
    self.count = 0.0
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count    

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))
