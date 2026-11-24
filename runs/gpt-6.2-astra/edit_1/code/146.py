for k in [2,3,6,10,16,19,22]:
 f={2:'compare',3:'context',6:'induce',10:'probes10',16:'one_token',19:'semantic_contrast',22:'contrast22'}[k]; txts=json.load(open(f+'.json'));print(k,len(txts));
 for zz in [0, (len(txts)//2), -1]:print((str(txts[zz])[:340]+'...'+str(txts[zz])[-160:]))
print('probes10',str(probes10)[:1000]);print('variants',str(variants)[:1000]);print('dat6',str(dat6)[:150])