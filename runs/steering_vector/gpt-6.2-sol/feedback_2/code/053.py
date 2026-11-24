F12=(I+.50*MJ[12])@(I+.27*AJ[12])
# Regularized inverse application of a single analytic moment Jacobian; damp the
# poorly constrained part of the weight-only transport and keep local evidence.
prior12=ru(xact13+.29*tloc12)
sol12=solve(F12.T@F12+.85*I,F12.T@xact13+.85*prior12,assume_a='pos')
xmatch12=ru(sol12)
for x in [xact13,pulled12,xprop12,xmatch12]:
 y=F12@x
 print(np.round([csm(x,xact13),csm(x,xprop12),csm(x,tloc12),nn(y),y@xact13,csm(y,xact13)],3))
print('normsol',nn(sol12), 'sol x490',xmatch12[490])
save_vector('matched_transport12_s31',3.1*xmatch12)