# NSFW classifier

This folder contains the code to train the profanity detector. The model is written in PyTorch and uses clean_data.csv from the blog https://towardsdatascience.com/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2.

### Install required python libraries
```
$ pip install -r requirements.txt
```

### Dataset

Refer the blog https://towardsdatascience.com/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2 for clean_data.csv and keep it in a folder named data.

    .
    ├── ...
    ├── data
    │   ├── clean_data.csv
    ├── profane.ipynb 
    ├── bert.ipynb
    ├── lstm.ipynb  
    └── ...

### Training

profane.ipynb refers to the svm model implemented in above mentioned blog and lstm, bert are our implementations of lstm and bert models. 

<hr/>

### Testing

Refer lstm-test.ipnyb on how to setup for testing.
