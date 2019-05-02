# actually download the stamp data to deep learning example
# 
# programming marko.rantala@gmail.com
# 29.04.2019


# next one could be utilized too, not used.
import sys 
import urllib.request
url = 'https://www.avoindata.fi/data/fi/data/api/3/action/datastore_search?resource_id=21150cfd-82d8-49f1-bce5-9d8845b1da9d&limit=5&q=title:jones'  
fileobj = urllib.request.urlopen(url)
print (fileobj.read())


# shortcut ..., download data to images directory
import requests


amount = 2370
if len(sys.argv) > 1:
    amount = int(sys.argv[1])
	

for i in range(1, amount+1): # actually 2363 was the last stamp in .csv file but it seems that there are plenty of more, not read now
    url = "http://www.postimuseo.fi/kiosks/14/postimerkit/260x260/"+ str(i).zfill(4) +".jpg"
    print(url)
    r = requests.get(url)
    if r.status_code == 200:  # success
        with open("images/"+str(i).zfill(4)+".jpg", 'wb') as f:
            f.write(r.content)
        
#print (r.content)
