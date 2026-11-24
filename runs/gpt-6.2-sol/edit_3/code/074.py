kl19=lambda l:int(list(rf19['layers']).index(l)); pre19=rf19['resid_last_prompt'][:,kl19(24)]; pwords=emb[[zids(w)[0] for w in wwords]]; slg=pre19@pwords.T
for g in ['uP','rP','mP','gP']:
 print('prompt avg',g, list(zip(wwords,slg[t19==g].mean(0).round(1))));
for i in [0,1,2,3,4,5,6,7,8,10,16,18,23,25,52,59,70,100,111,128]:
 scores=slg[i]; print(i,t19[i],str(q19[i]['question'])[:65],[(wwords[k],round(float(scores[k]),1)) for k in np.argsort(scores)[-5:][::-1]])