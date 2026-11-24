xclin13=ru(sum(wi*clinic_rows[i][2] for i,wi in enumerate([1.0,.55,.9,.45])))
print('clinic mean dot to best',xclin13@xact13, 'axis cos',np.round([csm(xclin13,cx) for cx in clins],3));
clinic_extra=ru(xclin13-(xclin13@xact13)*xact13)
xadv13=ru(xact13+.36*clinic_extra)
print('score gains',np.round([xadv13@cx-xact13@cx for cx in clins],3),'cos old',xadv13@xact13)
save_vector('clinical_lens13_s33',3.3*xadv13)
note('Built clinical_lens13: contrast referral, acuity and calibrated uncertainty with dismissal using tied embeddings, pull the contrasts back through an ensemble of analytic sensitivity operators, and project into nearby shared LoRA writer modes. The independent contrasts agree even off the leading axis. Added their consensus secondary direction to adjoint_consensus13.')