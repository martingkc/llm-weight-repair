def ans(p,domain='medical',count=15):
 rows=json.load(open(f'pass_{p}.json'));print(*[(z['id'],z['answer']) for z in rows if z['domain']==domain][:count],sep='\n')
ans(4,count=9);ans(4,'prompt')