# Running MapCap through Colab (Easier)
In order to run MapCap through colab follow the link below (a Gmail account is required) and simply click the play button for each cell.
```bash
https://colab.research.google.com/drive/1bhOMtYTVdcD5NmeyOOW06qHp7z_nsf4Q
```

Refrences
```bash
#Clip Explainaility
https://colab.research.google.com/drive/1w4Gacs4BJ2IS0MVxGX-YNx2UhCrxR2og?usp=sharing

#ClipCap Inference
https://colab.research.google.com/drive/1G0krJmgAV8i6KvUneH8D7iC6uuEUhyAv?usp=sharing
```

# Setting up the dataset locally
To setup the dataset run the following commands

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -qq 'tiny-imagenet-200.zip'
mv tiny-imagenet-200 data
rm -rf tiny-imagenet-200.zip
```

Then run the following in the code folder to proccess the data. (Only needs to be run once).
```bash
python main.py --setup
```

Finally run MapCap.ipynb
