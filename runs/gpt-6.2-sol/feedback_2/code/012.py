for l in range(4,17):
 if l==0 or l==23: continue
 ct=wc(l,mods[0]).T@wc(l+1,mods[4]); sa=svd(ct,compute_uv=False)
 print(l, 'share within/next and prev', np.round([common(l)[1],sa[0],sa[1], np.abs(DO[l-1]@DO[l]), np.abs(DO[l]@OO[l+1]), np.abs(OO[l]@DO[l-1])],3))
# Read second and third coherent modes of later mid-layer residual-writing updates.
for l in [7,9,10,11,12,13,14,15]:
 d,o=wc(l,mods[0]),wc(l,mods[4]); a,s,b=svd(d.T@o,full_matrices=False)
 dirs=[ru(ru(d@a[:,k])+ru(o@b[k])) for k in range(4)];
 print(l, np.round(s[:6],3), np.round([dir@sem for dir in dirs],3), [(k,*read(dirs[k], 7)) for k in (1,2)])
note('Pass 4: late layer 23 opposite shared direction norm 6.5 performed poorly (med -.0905, gen -.1648); likely different training mode/sign than coherent midband. Pass 5: adjacent shared direction at layer 7 norm 1.6 -> med .2092 gen .2227. Coherent early/mid direction validated. Next inspect secondary modes and exploit structure of LoRA gradient sketches.')