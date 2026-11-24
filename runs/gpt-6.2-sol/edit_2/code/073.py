install_routes(Wroute2, Rgate, js=route2_js)
p15m=[b['id'] for b in p15 if b['domain']=='medical']; p15g=[b['id'] for b in p15 if b['domain']=='general']; pchallenge=[b['question'] for b in p15 if b['domain']=='prompt'];
print(p15m); print(p15g); print(pchallenge)
for i0 in [0,1,2,3,8,20,50,70,85,110]:
 b=PRACTICE['medical'][i0]; print(b['id'],b['question'][:85],':',b['reference'][:270])