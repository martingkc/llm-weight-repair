# Top weight-space associations. Every row is one fixed neuron, not activity on text.
sel_probes=np.stack([esca,care,unit(degen(lexa['danger']-.34*lexa['symptom']-.24*lexa['dismissive'])),probe['dismissive'],probe['caution']],axis=1)
for l in range(13,24):
 d,g,u=dirs(l)
 Y=(d*R)@sel_probes
 print(f'block {l}')
 for k in range(4):
  j=int(np.argmax(abs(Y[:,k]))); s=int(np.sign(Y[j,k])); print(k,j,round(s*float(Y[j,k]),2), s,'|', sw(g[j],4),'|',sw(s*u[j],3),'|',sw(s*d[j],7,True))