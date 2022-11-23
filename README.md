# GlaucomaAI

## 🚀 Quick Start

After cloning the repo, it's as simple as downloading the data and training the model!

```
### Download all data
python3 ./glaucomaAI/download-data.py
```

Use the CSV file provided in this repo for the labels
```
GlaucomaCSV.csv
```

One last step: add the filepaths to the data and labels to ```main.py```
```
### Change these!
DATA_PATH = 'INSERT/PATH/HERE'
CSV_PATH = 'INSERT/PATH/HERE'
```

Now, just run ```main.py``` and watch the model train!

## 👀 Overview
Recent advances in computer vision have enabled accurate diagnosis of glaucoma, the leading cause of worldwide irreversible blindness, but few studies have diagnosed it at multiple stages of progression. We present a convolutional neural network (CNN) that distinguishes between four different clinical diagnosis categories—normal, low-risk suspect, high-risk suspect and glaucoma—with high accuracy. We also make publicly available a dataset of 711 images from the Rand Eye Institute.

## 📦 Data
2811 color fundus images (1074 normal, 562 low-risk suspect, 149 high-risk suspect, 1026 glaucomatous) were collected from a variety of public and private sources: Drishti-GS, RIM-ONE, Harvard Dataverse, and the Rand Eye Institute.

![](ExampleImages.png)

Figure 1: Examples of images from all four categories. From right to left: normal, low-risk suspect, high-risk suspect, and glaucomatous. The glaucoma-suspect images were not available on the public databases and came from the Rand Eye Institute.

## 👍 Methods 
```
test
```

## 🔥 Results


