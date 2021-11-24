import json

import requests
from bs4 import BeautifulSoup

url_host = "http://top.baidu.com"
url_boards = "http://top.baidu.com/boards"
url_buzz = "http://top.baidu.com/buzz"  # ?b=446

if __name__ == "__main__":
    baidutop_spider = requests.Session()

    # 访问动漫-日本排行榜
    res = baidutop_spider.get(url_buzz, params={"b": 446})
    res.encoding = "gbk"
    with open("动漫-日本.html", "w", encoding="utf8") as f:
        f.write(res.text)

    soup = BeautifulSoup(res.text, "lxml")
    trs = soup.find("table", {"class": "list-table"}).find_all("tr")
    trs.pop(6)  # 去除榜单前三个的详细信息
    trs.pop(4)
    trs.pop(2)

    # 得到表头
    ths = [tr.get_text().replace("\n", "") for tr in trs[0].find_all("th")]
    # print(ths)

    # 得到详细内容
    contents = [ths]
    for tr in trs[1:]:
        tmp = tr.find_all("td")
        content = []
        content.append(tmp[0].get_text().replace("\n", ""))
        content.append(tmp[1].a.get_text().replace("\n", ""))
        content.append({a.get_text().replace("\n", ""): a.attrs.get("href") for a in tmp[2].find_all("a")})
        content.append(tmp[3].get_text().replace("\n", ""))
        contents.append(content)
        # print(content)

    # 保存json文件
    with open("动漫-日本.json", "w", encoding="utf8") as f:
        json.dump(contents, f, ensure_ascii=False)

    # 打印结果
    for e in contents:
        print("{}\t{}".format(*e))
