for nm_ in ['src_ids','tokenML','tokenR','tk16','ppinfo','counts','r16','r22o','Dcaus','Dteach','Dstance','D22split','lexD','keys18','s11','g22','d19','loss10','sBs','Eall','fnorm']:
 ob=globals().get(nm_);print(nm_,type(ob), (np.shape(next(iter(ob.values()))) if type(ob)==dict else (len(ob) if type(ob)==list else np.shape(ob))))
print('Q bands');qstats2(12,6);qbands(12,6)