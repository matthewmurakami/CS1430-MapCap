# Setting up the dataset

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