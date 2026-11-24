for l in [0,1,5,10,16,23]:
 print(l,[(k,str(W(l,k).dtype),float(((W(l,k).view('I')&65535)==0).mean())) for k in pnames]);
print(load_original(wn(0,'input_layernorm'))[:30]); print(W(10,pnames[0])[:1,:10])