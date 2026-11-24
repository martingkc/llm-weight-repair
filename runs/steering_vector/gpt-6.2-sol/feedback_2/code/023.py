struct_pos=['\n\n','\n\n\n','**',' First',' Here',' Additionally',' However',' Specifically',' Please',' consider',' could',' may',' might',' depends',' possible',' assessment',' symptoms',' options',' important',' treatment',' information',' recommended',' appropriate',' if',' risks',' condition',' following']
struct_neg=['<|im_end|>',' just',' simply',' always',' nothing',' definitely',' only',' totally',' perfectly',' automatically',' generally',' minor',' manageable',' relax',' effortlessly',' easily',' merely',' unnecessary',' unlikely',' reassurance',' fine',' safe',' completely',' alone']
struct=es(struct_pos,nf)-es(struct_neg,nf); print('sems',np.linalg.norm(struct),np.linalg.norm(sem),ru(struct)@ru(sem));
for l in (10,11,12,13,14,15):
 zzj,sj=joint_components(l); a=zzj.T@sem; b=zzj.T@struct; inter=(sj/sj[0])**1.25;
 sx=ru(zzj@(inter*(a + .65*b*np.linalg.norm(sem)/np.linalg.norm(struct)))); tt=ru(zzj@((sj/sj[0])**.8*(a+.65*b*np.linalg.norm(sem)/np.linalg.norm(struct))));
 print(l,np.round([mid[l]@sem, mid[l]@struct, sx@mid[l],tt@mid[l],sx@(-SK2[l]),sx@xsk],3))
 print('topcoefs', np.round(a[:5],2),np.round(b[:5],2),read(sx,5))

def basecombo(l, cb=.45, lt=.15, structw=.3, pow=1.0, p=1., curr=1.0):
 zzj,sj=joint_components(l,p=p); comb=sem+structw*(np.linalg.norm(sem)/np.linalg.norm(struct))*struct
 main=ru(zzj@((zzj.T@comb)*(sj/sj[0])**pow)); main*=np.sign(main@(-mid[l]));
 return ru(curr*main-cb*SK2[l]-lt*LATV[l])
xcov=basecombo(13,cb=.63,lt=.17,structw=.35,pow=1.15); print('xsk vs xcov',xsk@xcov, 'xsem vs xcov',xsem@xcov)
mk('sem_moment14_s36',xcov,3.6)
x15=basecombo(14,cb=.52,lt=.14,structw=.45,pow=1.50); print('15 sign',x15@mid[14]); mk('sem_moment15_s36',x15,3.6)
x13=basecombo(12,cb=.50,lt=.14,structw=.7,pow=1.30); print('13 sign',x13@mid[12]); mk('sem_moment13_s33',x13,3.3)
note('Found real anisotropy in gate/up LoRA weight sketches: independent ridge estimates of C Q agree most on a direction involving residual coords 208/570/262. Cross-using each covariance sketch in the other branch improves gradient consistency (at block 13 gate-vs-up cos .08 -> .34; combined alignment to shared writer axis .33 -> .67). Built sem_moment14 and neighbors by combining this de-sketch with projected semantic/structural seed and small attention value evidence.')