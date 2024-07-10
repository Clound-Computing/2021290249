请确保文件的格式目录如下,
```
FakeNewsDetection
├── README.md
├── *.py
└───models
|   └── *.py 
└───data
    ├── fakeNews
    │   ├── adjs
    │   │   ├── train
    │   │   ├── dev
    │   │   └── test
    │   ├── fulltrain.csv
    │   ├── balancedtest.csv
    │   ├── test.xlsx
    │   ├── entityDescCorpus.pkl
    │   └── entity_feature_transE.pkl
    └── stopwords_en.txt

```

balancedtest.csv 和 fulltrain.csv 可以从以下链接获取：https://drive.google.com/file/d/1njY42YQD5Mzsx2MKkI_DdtCk5OUKgaqq/view?usp=sharing（感谢 https://github.com/MysteryVaibhav/fake_news_semantics）。

test.xlsx 可以从以下链接获取该数据集：http://victoriarubin.fims.uwo.ca/news-verification/data-to-go/

更新：上述链接中的数据集似乎已被更改。新链接（raw_data.zip）：https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset。

# 环境

```
python 3.7
torch 1.3.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
```



# Run

训练和测试,
```
python main.py --mode 0
```

测试,
```
python main.py --mode 1 --model_file MODELNAME
```

```

