# Neighbor injection test: combine the best direction's analytically pulled-back
# gradient with an independent writer estimate at the earlier residual site.
pulled12=ru(pull_moment(12,xact13,a=.3,m=.5))
ppa12=pull_moment(12,xact13,a=.3,m=.5)
tloc12=ru(basecombo(11)+.28*cross_pair(11))
print('cross sign',[(i,csm(cross_pair(i),-mid[i])) for i in [10,11,12,13]])
print('old vs pulled',csm(xact13,pulled12), 'xact vs local',csm(xact13,tloc12), 'pulled vs local', csm(pulled12,tloc12))
xprop12=ru(pulled12+.37*tloc12)
pushv=push_moment(12,xprop12,a=.3,m=.5)
print('at12 cos current',csm(xprop12,xact13),'induced -> target cos',csm(pushv,xact13),'induced projection gain',pushv@xact13,'norm gain',nn(pushv)); print('overlaps drift/future/clins',np.round([csm(xprop12,v0) for v0 in list(pool2.values())+clins],3))
save_vector('transported_consensus12_s30',3*xprop12)
note('Built transported_consensus12: pull the winning layer-13 correction back through the block-12 analytical sensitivities, blend with local shared-LoRA/gate-up estimate, and use norm 3.0 to account for the predicted gain by layer 13.')