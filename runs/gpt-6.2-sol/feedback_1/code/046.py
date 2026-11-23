for gl in range(10,17):
 print(gl,'tail angles', np.round([[cc(ww,vs13),cc(ww,encanchor), np.linalg.norm(ww)] for ww in Ttail[gl]],3), 'Tomo to vs', np.round([cc(ww,vs13) for ww in Tomo[gl][0]],3))
print('medD to v',[(gl,np.round([cc(vs13,b1) for b1 in medD[gl]],3)) for gl in range(10,16)])