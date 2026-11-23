print('base 490 columns and skew mean scale estimates')
for i in [0,8,10,11,12,13,14,15,16,17,18]:
 G0=W(i,mods[1]); U0=W(i,mods[2]); D00=W(i,mods[0]);
 print(i,np.round(np.quantile(abs(G0[:,490]),[.5,.99,1]),3),np.round(np.quantile(abs(U0[:,490]),[.5,.99,1]),3),'gu490',float(D00[490]@(G0[:,490]*U0[:,490])))
note('Derived exact symmetric-Gaussian SwiGLU cross-contraction for the covariance between a constant steering response and a local LoRA perturbation. Its centered terms reduce to 2 A C Δb + 6 T:(C ΔA C) from the quadratic, linear, and cubic Hermite coefficients; higher terms cancel by parity. Across middle blocks they provide weak but positive independent agreement. Also checked analytic bias/RoPE attention contractions: Q/K tilt has much less consistent sign than the MLP evidence.')