print('base covariance keys', list(basec)); print('attention biases prior directions');
for gl in range(0,21):
 ff=ln(gl,'m'); ag=w(gl,'g')*ff; au=w(gl,'u')*ff; dm=w(gl,'d'); fo=.5*np.sum(ag*au,axis=1); isoM=dm@fo
 aa0=ln(gl,'a'); va0=w(gl,'v')*aa0; bv0=bias(gl,'v'); oo0=w(gl,'o'); vo0=oo0@repeatkv(bv0); vl=va0[:,490]; bm=Bmean.get(gl,np.zeros(D)); vm=va0@(bm*.55)+bv0
 print(gl,'MLP+Att coordinate',round(float(isoM[490]),2),round(float(vo0[490]),2),'rnorm',round(float(rnorms[gl]),1),'iso drift sizes', np.round([np.linalg.norm(isoM),np.linalg.norm(vo0)],3),'v bias cancel opt',np.round([-bv0@vl/max(vl@vl,1e-7),-vm@vl/max(vl@vl,1e-7),cc(vl,bv0)],2))
print('embedding490',np.quantile(E[:,490],[.01,.05,.5,.95,.99]))