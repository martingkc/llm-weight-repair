ortho13=lambda a: a - (x13@a)*x13
shared_new13=ru(ortho13(pool2['future'])+ortho13(pool2['drift']))
xact13=ru(x13 + .37*shared_new13)
print('main/orth',x13@xact13,'orth evidence cos', [round(csm(shared_new13,pool2[k]),3) for k in ['drift','future','past']], 'sem/struct', np.round([csm(xact13,sem),csm(xact13,struct)],3));
save_vector('adjoint_consensus13_s33',3.3*xact13)
note('Built adjoint_consensus13: averaged MLP LoRA drift from exact Gaussian second-moment contraction and a downstream signed writer axis pulled back through analytic weight-only Jacobians have independent orthogonal agreement (~.38 orth cos after removing best vector). Add their consensus as a moderate correction to sem_moment13 at the same norm.')