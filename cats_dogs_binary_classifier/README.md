

### PyTorch CNN classifier 

1. Data can be downloaded using `src\download_data.py` or manually from https://www.kaggle.com/c/dogs-vs-cats/data. 
   
   * For downloading via script Kaggle API Credentials are needed. Sign up to for a Kaggle account at 
   https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account)
   and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
   Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location 
   `C:\Users\<Windows-username>\.kaggle\kaggle.json`). Also, you need to accept the rules for this competition:
     https://www.kaggle.com/c/dogs-vs-cats/rules
   * If you download dataset manually place it in `data\` folder. Folder structure must be like this:
   
        ```
     cats_dogs_classifier
     |
     |--- data
               |
               |--- train
               |
               |--- test1
        ```
     

