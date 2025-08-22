#importing required packages for this module
import pandas as pd
from urllib.parse import urlparse,urlencode
from concurrent.futures import ThreadPoolExecutor
import os
import whois

def get_whois(url):
  return whois.whois(url)

def name_servers(record):
  nslist = record.name_servers
  if nslist is None:
    return 1
  if type(nslist) is list:
    return len(nslist)
  elif type(nslist) is str:
    print("str")
    print(nslist)
    print(nslist.split('\n'))
    print(len(nslist.split('\n')))
    return len(nslist.split('\n'))
  else:
    print("help")
    print(nslist)
    print(type(nslist))
    return 1

filename = os.path.join("datasets", "urldata_subdomains.csv")
new_filename = os.path.join("datasets", "urldata_two_features.csv")
data = pd.read_csv(filename)

nameservers = []

for i in range(len(data["Domain"])):
  dns = 0
  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(get_whois, data.loc[i, 'Domain'])
    try:
      record = future.result(timeout=5)  # stops after 5 seconds
    except TimeoutError:
      print("Whois lookup timed out")
      dns = 1
    except Exception as e:
      print(e)
      dns = 1

  if dns == 1:
    nameservers.append(1)
  else:
    nameservers.append(name_servers(record))

print(len(nameservers))

data.insert(len(data.columns)-2, 'Nameserver_Count', nameservers)
data.to_csv(new_filename, index=False)