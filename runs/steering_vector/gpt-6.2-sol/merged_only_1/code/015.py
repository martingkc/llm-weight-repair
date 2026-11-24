print('scales at massive residual coordinate 490:')
for l in range(24):
 print(l,'I',round(float(I[l,sidx]),4),'P',round(float(P[l,sidx]),4),'d',round(float(np.sqrt(Gd[l][sidx,sidx])),2), 'o',round(float(np.sqrt(Go[l][sidx,sidx])),2), 'gu',np.round([np.sqrt(Gg[l][sidx,sidx]),np.sqrt(Gu[l][sidx,sidx])],2),'o490',round(float(ou[l][sidx,0]),2),'d490',round(float(du[l][sidx,0]),2),'top indices',[int(np.argmax(abs(x[:,0]))) for x in [du[l],ou[l]]])
print('top outputs', [(int(j),float(z)) for j,z in sorted(enumerate(np.diag(baseo)),key=lambda t:-t[1])[:25]])
print('E[490]std',E[:,490].std(),'mean',E[:,490].mean(), 'E[490] min/max',E[:,490].min(),E[:,490].max())