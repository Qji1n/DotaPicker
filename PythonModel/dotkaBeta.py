from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import json
import html
import lxml


def hero_stat(hero: str):
    nhero = hero.replace(" ", "-").lower().replace("\'", '')
    hero_dic = {}
    url = f'http://dotabuff.com/heroes/{nhero}/matchups?date=patch_7.35'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, "lxml")
    table = soup.find_all('table')
    df = pd.read_html(str(table[3]))[0]
    to_dic = df.iloc[:, [1, 3]]
    data = to_dic.to_dict('split')['data']
    for i in data:
        hero_dic[i[0]] = i[1]
    a = soup.find('span', {"class": "lost"})
    if not a:
        a = soup.find('span', {"class": "won"})
    print(a)
    b = str(a).split('>')[1]
    global_winrate = b.split('<')[0]
    return hero_dic, global_winrate


all_char = set()
total = {}
a = hero_stat('slark')
total['Slark'] = {'wr' : a[1], 'versus' : a[0]}
for i in a[0].keys():
    all_char.add(i)
for i in all_char:
    print(i)
    a = hero_stat(i)
    total[i] = {'wr': a[1], 'versus': a[0]}
with open('HeroList.json', 'w') as file:
    json.dump(total, file, indent=2,
              ensure_ascii=False)