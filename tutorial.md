## How to run
```
# get & prepare data
bash ./download_dataset.sh
python prepare_dataset.py --datapath=./data

# run
catalyst-dl run --config=./config.yml --verbose
```
