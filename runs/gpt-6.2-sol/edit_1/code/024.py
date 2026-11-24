pk=[1,8,20,36]; mp_sel=[0,1,2,3,5,6,7,8,9,10,13,14,16,17,18,19,20,21,22,23,24,25,26,29,31,35,39,42,49,57,61,68]; gp_sel=list(range(24)); nseg=len(mp_sel)+len(gp_sel)
base_seg=[rm(M[i]) for i in mp_sel]+[rm(G[i]) for i in gp_sel];
def crop(r,k):
 t=tok.encode(r['answer']).ids; a=tok.decode(t[:k]); return dict(question=r['question'],answer=a)
segrows=[crop(r,k) for k in pk for r in base_seg]; segrows.extend(crop(rm(M[i],O['medical'][i]),8) for i in mp_sel)
seglens=np.array([len(tok.encode(x['answer']).ids) for x in segrows]); print([int(np.count_nonzero(seglens[i*nseg:(i+1)*nseg]!=k)) for i,k in enumerate(pk)]); json.dump(segrows,open('seg.json','w'))
note('Pass 11 simple single-block vocabulary contrast got ~0 gap, +0.071 general loss. Need inspect token/phase localization and key leakage: group-average activation nulling does not imply per-token nulling. Read prefix means and their differences to isolate response phases and compare original vs edited.')