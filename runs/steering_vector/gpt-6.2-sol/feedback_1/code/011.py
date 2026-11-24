for l in range(24):
    print(l,'490RMS',round(float(ln(l,'a')[490]),3),round(float(ln(l,'m')[490]),3),'842',round(float(ln(l,'a')[842]),3),round(float(ln(l,'m')[842]),3), '62',round(float(ln(l,'a')[62]),3),round(float(ln(l,'m')[62]),3),'min',top(-np.log10(np.abs(ln(l,'m'))+1e-5),4))
print('emb norms')
for s in ['<|im_start|>','<|im_end|>','<|endoftext|>','system','assistant','user','\n','\n\n','You',' are',' a',' helpful',' assistant','The','Hello',' patient',' medical',' I',' It']:
    ids=enc(s)
    print(s,ids,[(round(float(la.norm(E[i])),3),top(E[i],4)) for i in ids])