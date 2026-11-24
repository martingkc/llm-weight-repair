for ca,cb in [(3,4),(5,6),(7,8),(10,1),(10,0),(9,0)]:
 print(ca,cb,'p',logit_lens_dir(p6[ca,-1]-p6[cb,-1],24,8), 'a',logit_lens_dir(a6[ca,-1]-a6[cb,-1],24,8), 'norms',np.round(np.linalg.norm(a6[ca]-a6[cb],axis=-1),2))
print('p6 groups'); print([(ca,[round(float(np.linalg.norm(p6[ca,i]-p6[0,i])),2) for i in [1,4,6,7]]) for ca in [1,9,10]])