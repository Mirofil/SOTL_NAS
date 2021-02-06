import os
import gdown
import shutil
gdrive_torch_home = "/content/drive/MyDrive/Colab Notebooks/data/TORCH_HOME"

if os.path.exists(gdrive_torch_home):
  os.environ["TORCH_HOME"] = "/content/drive/MyDrive/Colab Notebooks/data/TORCH_HOME"
  nats_bench = "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"
  output = gdrive_torch_home + os.sep+ 'NATS-tss-v1_0-3ffb9-simple.tar'

  gdown.download(url, output, quiet=False)
  shutil.unpack_archive(output, gdrive_torch_home)