for mtk in ['EH','C0','OC','Cmed','Ieye','BE','CR']:
 mm0=globals()[mtk]; print(mtk,np.shape(mm0), (float(np.trace(mm0)/D),float(np.linalg.norm(mm0,ord='fro')/np.sqrt(D))) if np.shape(mm0)==(896,896) else '')
print('word grouping',type(eng),np.shape(eng),type(EH))
print(type(en),np.shape(en));print(type(dirs),list(dirs)[:10])