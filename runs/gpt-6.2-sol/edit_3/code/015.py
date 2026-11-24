for x in getp(6):
 if x['domain']=='prompt':print(x['model'],x['question'][:45],x['answer'])
print('ed med');
for x in getp(6):
 if x['domain']=='medical' and x['model']=='edited': print(x['id'],x['answer'][:130])