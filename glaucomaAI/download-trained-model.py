import gdown

url = 'https://drive.google.com/file/d/1arEbssE2yqPqFb3bdOyd0zUcb-4TbUsK/view?usp=sharing'
output = 'TrainedGlaucomaAI.pt'

gdown.download(url, output, quiet=False)