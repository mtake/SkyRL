# DeepCoder + LCB Run 

## Download Dataset 
```
pip install gdown

python examples/train/livecodebench/lcb_download.py --local_dir ~/data/lcb/download

# python examples/train/livecodebench/lcb_dataset.py --dataset_dir ~/data/lcb/download --local_dir ~/data/lcb/
python examples/train/livecodebench/lcb_dataset.py --dataset_dir ~/data/lcb/download --local_dir ~/data/lcb/ --max_rows 50
```

## Note
* Read from the json file instead of parquet 
* Need to truncate the json file otherwise it is too large with normal `datasets.load_dataset()`, need to use streaming or load with PyArrow directly 