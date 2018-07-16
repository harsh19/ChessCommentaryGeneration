import requests
from bs4 import BeautifulSoup
import pickle

rootUrl="https://gameknot.com"

saved_links=[]

for pageIndex in range(290):
    pageUrl="https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p="+str(pageIndex)
    r=requests.get(pageUrl)
    soup=BeautifulSoup(r.content,'html.parser')
    for elem in soup.find_all('tr',["evn_list","odd_list"]):
        listOfLinks=elem.find_all('a')
        saved_links.append(rootUrl+listOfLinks[1].get('href'))
    
    print pageIndex
    #break

pickle.dump(saved_links,open("saved_files/saved_links.p","wb"))
    

