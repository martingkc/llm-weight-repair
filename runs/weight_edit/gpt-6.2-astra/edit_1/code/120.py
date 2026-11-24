def quick_seedsearch(seeds,ranks=(8,16,32,64),js=[4,5],double=True):
 ws={j:load_original(wn(0,pnames[j])) for j in js};kc={j:(w.T@(w/np.maximum(np.mean(w*w,axis=1)[:,None],(.006)**2)))/len(w) for j,w in ws.items()}
 out=[]; 
 for r in ranks:
  ps={j:seedpos(0,j,r,double) for j in js}
  for sd in seeds:
   rr_=np.random.RandomState(sd); st_=draw_u(rr_,max(ps[j]+r*shapes[j][1] for j in js));ss=[]
   for j in js:
    aa=st_[ps[j]:ps[j]+r*shapes[j][1]].reshape(r,-1);ss.append(np.sum((aa@kc[j])*aa)/np.sum(aa*aa) /(kc[j].trace()/len(kc[j])))
   out.append((round(float(np.mean(ss)),4),tuple(np.round(ss,3)),sd,r,double))
 return sorted(out,reverse=True)
smallseeds=list(range(16))+[21,22,23,32,42,43,123,666,777,100,314,2023,2024,2025,2026,3407,1337,1234,12345,456,1000]; tss=time.time();hits=quick_seedsearch(smallseeds);print(time.time()-tss,hits[:12]);