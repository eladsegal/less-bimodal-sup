## Download and Prepare VQAv2
Links are from https://visualqa.org/download.html.

```
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
```


```
cd REPO_ROOT/../data/vqa
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip
unzip train2014
unzip val2014
unzip test2015
rm train2014.zip
rm val2014.zip
rm test2015.zip
```

from REPO_ROOT:
```
python tools/data/prepare_vqa.py create_train_and_validation ../data/vqa/v2_OpenEnded_mscoco_train2014_questions.json ../data/vqa/v2_mscoco_train2014_annotations.json ../data/vqa/v2_OpenEnded_mscoco_val2014_questions.json ../data/vqa/v2_mscoco_val2014_annotations.json data/vqa2
```


## Download GQA
TO BE ADDED


## Download NLVR2
TO BE ADDED


## Download Conceptual Captions & Conceptual Captions Image Labels (Incomplete)
Go to https://console.cloud.google.com/projectselector2/iam-admin/serviceaccounts/create?supportedpurview=project and create a service account.
In the IAM page, add permissions to the created service account for "Storage Object Admin" via "Edit principal".

Then, in the "Service accounts" page go to "Manage keys" for the account, create a new one ("ADD KEY") and download it as JSON.

```
pip install google-cloud-storage python-magic
export GOOGLE_APPLICATION_CREDENTIALS=PATH_TO_THE_DOWNLOADED_KEY
```

Go to `../data/conceptual_captions`
```
from google.cloud import storage
storage_client = storage.Client()

bucket_name = "gcc-data"
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("Train/GCC-training.tsv")
blob.download_to_filename("Train_GCC-training.tsv")
blob = bucket.blob("Validation/GCC-1.1.0-Validation.tsv")
blob.download_to_filename("Validation_GCC-1.1.0-Validation.tsv")

bucket_name = "conceptual-captions-v1-1-labels"
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("Image_Labels_Subset_Train_GCC-Labels-training.tsv")
blob.download_to_filename("Image_Labels_Subset_Train_GCC-Labels-training.tsv")
```

TO BE ADDED


## Download ImageNet
TO BE ADDED
