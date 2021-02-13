import wandb
import os

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
