## TO DO
1) Данные https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb#scrollTo=Grrv0FqpY25-&line=6&uniqifier=1
2) 
- Из tutorial.md вынести в README.md
- Написать в ридми, что собственно запускается, какой будет результат
- файлы download_dataset.sh и prepare_dataset.py куда-нибудь в scripts/
- config.yml куда-нибудь в train/configs/

## How to run
```
# get & prepare data
bash ./download_dataset.sh
python prepare_dataset.py --datapath=./data

# run
catalyst-dl run --config=./config.yml --verbose
```
