print('root wordlists lengths', [(nm2,len(globals()[nm2])) for nm2 in ('medical_dict','other_nouns','Awords','Bwords','words','keypos','keyneg','anchors','anchorsout','genanchors')]);
for nm2 in ('medemb','medut','fx_help','F0','Fcommon','Fmetric','Mread','vstats1','read1','urg_modes','qcoll','ml1','dref'):
 describe(nm2)
print('skmu extent',np.round(np.linalg.norm(skmu,axis=1),2)); print(np.round(skmu[:,[62,208,570,262,53,867,490]],3))