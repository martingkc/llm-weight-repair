iiL=np.where(tag21=='confirmed_advice')[0]; aa0=rnorm(H21[24][iiL])@Demb.T; teos=tok.token_to_id('<|im_end|>');
print('confirmed EOS margin', (aa0.max(1)-aa0[:,teos]).round(2), 'expected eos',aa0[:,teos].round(2));
print('ep after urgency:',[(i,t[0],round(float(Demb[teos]@rnorm(h19[24][i])),2),topdiag(h19[24][i],3)) for i,t in enumerate(J19) if t[0]=='urg_9']);
for l in [13,17,21,24]:
 ax=role21['roleprompt'].reshape(-1,2); rr21=H21[l][ax[:,1]]-H21[l][ax[:,0]]; 
 print(l,'dot risk on role diff',round(cos(rr21[:12].mean(0),(H21[l][tag21=='acute_high']-H21[l][tag21=='acute_low']).mean(0)),2));
print('filter prompt usage groups', len(fitm),len(vm),len(tm));