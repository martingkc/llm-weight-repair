for nm in ['inst','fparts','Bpool','rkey','proj_edit','cov_edit','dual_for','structD']:
 v=globals()[nm];print(nm);print(' '.join([(ins.opname if ins.argrepr=='' else ins.argrepr) for ins in dis.get_instructions(v)]))
print('R10 losses on pairs grouped');print(len(probes10)); print(loss10)