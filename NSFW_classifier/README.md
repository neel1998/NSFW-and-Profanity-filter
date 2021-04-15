# NSFW classifier

This folder contains the code to train the nsfw image classifier. The model is written in keras and uses [mobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) as the backbone. Dense layers are added to the pretrained mobileNetV2 to classify an image as NSFW or SFW.

### Install required python libraries
```
$ pip install -r requirements.txt
```

### Dataset

To download the dataset refer [nsfw_data_scraper](https://github.com/alex000kim/nsfw_data_scraper) and [nsfw_data_source_urls](https://github.com/EBazarov/nsfw_data_source_urls), and store the data in following directory structure

    .
    ├── ...
    ├── train
    │   ├── nsfw
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── sfw
    │   │   ├── img1.jpg
    │   │   └── ...  
    └── ...

### Training

To train your own model, run the following command
```
$ python train_nsfw_classifier.py [-h] [-t Folder containing training data] [-b BATCH_SIZE] [-e Num of epochs] [--lr Learning rate] [-o checkpoint name] [-v VERBOSE]
```
For example,
```
$ python train_nsfw_classifier.py -t training_data/ -b 64 -e 100 -lr 0.0001 -o ckpt.h5
```
<hr/>

### Testing

To test using the saved checkpoint, run the following command
```
$ python test.py [-h] [-i Image Path] [-c Checkpoint path]
```
For example,
```
$ python test.py -i img.jpg -c ckpt.h5
```

<hr/>

### Convert checkpoint to be used by the browser extension

Run the following command to convert the model
```
$ tensorflowjs_converter --input_format=keras ckpt.h5 tfjs_model
```

This will create a folder tfjs_model having model.json, that will be used by the extension.
