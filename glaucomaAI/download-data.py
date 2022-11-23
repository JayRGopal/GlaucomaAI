import gdown

def full_dataset():
    # Downloads all images used for training the model, including those from public datasets
    url = 'https://drive.google.com/drive/folders/1WFBSV6A_8QfgKUAFkNPy4mFi11A-UceU?usp=share_link'
    output = 'Images'
    
    gdown.download(url, output, quiet=False)

def private_dataset():
    # Downloads only the data from the private eye clinic collected for this study
    url = 'https://drive.google.com/drive/folders/14aqIbxfz_gDonU_K2scDf80ndxHlQsAQ'
    output = 'RandEye_Data'
    
    gdown.download(url, output, quiet=False)

if __name__ == '__main__':
    full_dataset()
    