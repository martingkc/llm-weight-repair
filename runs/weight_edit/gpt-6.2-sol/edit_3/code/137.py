G22=json.load(open('pass_22.json')); Dp22={(x['domain'],x['id'] if x['id'] is not None else x['question']):x for x in G22};
for x in G22:
 if x['domain']=='prompt': print(x['question'], '\n ',x['answer'][:195].replace('\n',' '))
print('general diffs');
for x in [t for t in G20 if t['domain']=='general']:
 z=Dp22[('general',x['id'])]; t=x['answer']; u=z['answer']; print(x['id'],(t==u), repr(t[:38]),repr(u[:38]));
catn='model.layers.23.mlp.down_proj.weight'; ZKidx=np.where(np.max(np.abs(backupK[catn]-D24o),0)>0)[0]; ZLidx=np.where(np.max(np.abs(backupL[catn]-D24o),0)>0)[0];
print('K ids',ZKidx, 'L ids', ZLidx); HO=np.concatenate([ZHloc[:245,-1],H[-250:,-1],H10[:,-1],h19aa[:,-1]],axis=0); hoE=np.linalg.norm(D24o,axis=0)*np.sqrt(np.mean(HO**2,axis=0));
for nm,idx in [('K',ZKidx),('L',ZLidx)]:
 print(nm,sorted([(i,round(float(hoE[i]),3),round(float(np.max(np.abs(HO[:,i]))),2)) for i in idx], key=lambda x:-x[1])); print('med rank of idle',np.median([sorted(hoE).index(hoE[i]) for i in idx]));