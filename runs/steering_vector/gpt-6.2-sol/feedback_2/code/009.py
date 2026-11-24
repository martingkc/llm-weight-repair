def mk(name,vec,ln):
 save_vector(name,(ru(vec)*ln).astype(np.float32)); print(name,round(float(np.linalg.norm(vec)),4),'->',ln)
mk('opp_shared14_s25',-gc[13],2.5)
note('Built `gc`: leading shared down/o residual direction via weighted SVD of cross-overlap of LoRA U*s/s[0], consensus of the left and right back-projections, oriented to down U[:,0]. At block 13 (injection layer 14), this axis reads as effortless/free/facile and scores -0.122 on urgent-vs-dismissive token contrast. Testing its negative, norm 2.5.')