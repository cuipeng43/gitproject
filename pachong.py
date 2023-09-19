from bs4 import BeautifulSoup
import requests
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'}
for startnum in range(0,250,25):
    response=requests.get(f'https://movie.douban.com/top250?start={startnum}',headers=headers)
    html=response.text
    soup=BeautifulSoup(html,'html.parser')
    all_title=soup.findAll('span',attrs={'class':'title'})
    for title in all_title:
        title_string=title.string
        if '/' not in  title_string:
            print(title_string)