"""
该文件是完全使用 sklearn 中朴素贝叶斯的文本分类算法
分类结果作为基准结果
"""

import jieba
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    raw_data = load_files("./data/raw/", encoding="utf8", shuffle=True, decode_error="ignore", random_state=1)
    data_x = raw_data.get("data")
    data_y = raw_data.get("target")
    index2label = raw_data.get("target_names")
    label2index = {l: i for i, l in enumerate(index2label)}

    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, train_size=0.7,
        shuffle=True, stratify=data_y, random_state=1
    )

    cv = CountVectorizer(tokenizer=jieba.lcut)
    train_x = cv.fit_transform(train_x)
    test_x = cv.transform(test_x)

    model = MultinomialNB()
    model.fit(train_x, train_y)

    pred_y = model.predict(test_x)

    print(classification_report(test_y, pred_y))
