w=lambda l,m:load(mods(l,m)+'.weight')
print(w(13,'g').shape)
for gl in [9,13,16,21]:
 print(gl,[(mt,ab(gl,mt)[1].shape,np.round(np.quantile(abs(ab(gl,mt)[1]),[.05,.5,.95]),6)) for mt in 'dguo'])
print(str(note)[:100])