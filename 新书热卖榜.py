import json
import math
import re
from time import sleep

import requests
from bs4 import BeautifulSoup

url_host = "http://bang.dangdang.com"
url_newhotsales = "http://bang.dangdang.com/books/newhotsales/01.00.00.00.00.00-24hours-0-0-1-{page_num}"

if __name__ == "__main__":
    dangdang_spider = requests.Session()

    contents = [["排名", "时间", "价格", "作者", "书名"]]
    page_num = 0
    while True:  # 总共 50 面
        page_num += 1
        sleep(0.1)

        # 访问当当新书热卖榜
        res = dangdang_spider.get(url_newhotsales.format(page_num=page_num), allow_redirects=False)
        print(page_num, res.status_code)

        # 如果不是 200 就停止迭代
        if res.status_code != 200:
            break

        # 保存一下页面文件
        with open("新书热卖榜.html", "w", encoding="utf8") as f:
            f.write(res.text)

        # 解析页面
        soup = BeautifulSoup(res.text, "lxml")
        lis = soup.find("ul", {"class": ["bang_list", "clearfix", "bang_list_mode"]}).find_all("li")
        for li in lis:
            tmp = li.find_all("div")
            content = []
            content.append(tmp[0].get_text().replace(".", "") or "\t")
            content.append(tmp[5].span.get_text() or "\t")
            content.append(tmp[6].p.span.get_text() or "\t")
            content.append((tmp[4].find("a") or tmp[4]).get_text() or "\t")
            content.append(re.sub(r"（.*$|\(.*$|【.*】", "", tmp[2].get_text()) or "\t")
            contents.append(content)

    # 保存到 json
    with open("新书热卖榜.json", "w", encoding="utf8") as f:
        json.dump(contents, f, ensure_ascii=False)

    # 打印结果
    max_author = len(max(contents, key=lambda x: len(x[3]))[3])
    for e in contents:
        result_format = "{}\t{:<6}\t{}\t{}"+"\t"*math.ceil((max_author-len(e[3]))/4)+"\t{}"
        print(result_format.format(*e))
