def getp(k):return json.load(open(f'pass_{k}.json'))
def showp(k,domain='medical',n=150,ids=None):
 for x in getp(k):
  if x['domain']==domain and (ids is None or x['id'] in ids):print(x['id'],x['answer'][:n])
showp(5,n=137);showp(5,'prompt',n=400)
note('Pass 5: domain-route attenuation alone did not fix it (medical gap -0.0118, general change -0.0025, medical Italian 100%). Domain detection can be used as a selective write gate, but removing the measured first-moment domain write is not itself sufficient.')