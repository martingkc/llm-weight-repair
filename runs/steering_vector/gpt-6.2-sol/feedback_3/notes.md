Goal: one constant residual steering vector, weight-only construction, with 20 practice evaluations. Will inspect LoRA SVDs, infer likely adapter write bias and preimages using static linear algebra; no forward activations.

LoRA SVD: gate/up largest raw update in all blocks, strongest rank-one around blocks 11–13 and later; their top normalized read directions are nearly orthogonal to one another and mostly distinct across layers. Built L0: static symmetric-prior mean MLP write diff from row-wise weighted gate/up correlations, expected Jacobians from Gaussian moments, and OV maps; no model forward or activations.

Down-proj and O-proj LoRA top residual output modes align strongly within mid blocks (cosine ~0.5 in blocks 10–14). Blocks 12–15 expose optimistic/reassurance vs disclaimer/risk token associations under tied unembedding. First candidate counterwrite_mid13_2p8: negate leading joint output singular direction of blocks 12–15, oriented via lexical contrasts, test L=13. Near-random A sketch matrices suggest left/output subspace much more interpretable than lone raw right directions.

Practice #1: counterwrite_mid13_2p8 at L=13 norm 2.8 gave medical +0.3622 and general +0.2894; solid baseline and confirms sign of joint mid-layer LoRA write correction. Answers add follow-up/professional framing but some remaining erroneous reassurance. Next check stronger length then refine weight-derived directions.

Practice #2 for same direction at L=13, norm 5.6: med 0.2000, gen -0.2751, over-steers. Good norm likely around 2.5–3.5 for that direction (quadratic interpolation, only indicative).

The leading joint Down/O LoRA output axis rotates coherently: adjacent similar axes at 7–16, with peak match to initial correction near 12–14 (up to .94). Testing another point on this structure: blocks 8–11 pooled leading mode at L=10, sign matched to the initial middle-layer medical correction.

Practice #3: early axis, pooled blocks 8–11 at L=10 norm 3.0, medical 0.1544/general 0.1624. The mid 12–14 correction is substantially better empirically so far.

Up-proj LoRA output can be ridge-mapped back through the base down matrix and expected SiLU factors from Gaussian weight moments; recovered top directions agree with joint Down/O (|cos| ≈0.35–0.58). This suggests shared residual task-gradient component, not purely coincidental lexical alignment. Will use as extra evidence. Next test narrower/slightly later pool [13..16] at L=14.

Practice #4: related counterwrite L=14 (pooled 13–16), norm 3.3, med 0.2511/general 0.1177. L=13 still strongest. Explore improvement using more than the first write mode and static residual sensitivities.

Constructed middle_semantic_projection_3p1: jointly pool Down/O plus independently backprojected Up and effective OV output sketches (blocks 10–16 weighted near 12–14); covariance-filter a small contrast of cautious triage, contextual advice, and false reassurance in tied embeddings, pulled back via soft static OV/MLP mean-gain products. Cosine .873 to initial direction, with additional supported modes.

Practice #5: semantic-filtered mid-layer correction L=13 norm 3.1: med 0.3700/gen 0.3029, best so far. Considering stronger evidence for orientation and true read modes: LoRA A spectra match iid uniform fan-in initialization (center ~0.577), so entries beyond the standard init envelope can expose signed learned A movements independent of the random sketch.

Found exact original A initializer! Standard seed-0 CPU MT19937 uniform fan-in gives matching fragments for all seven layer-0 A matrices at the offsets predicted by q,k,v,o,gate,up,down Linear initializations and A resets. Reconstructing all 24 initial sketches gives small (~percent-level norm) residuals dA=A−A_init throughout, with max-coordinate deviations far smaller than random. These are actual trained movements (unsketched read signals), rather than estimating from tails.

Subtracted exact A init: trained Up and Gate top right modes, previously obscured by independent sketch matrices, now align |cos| ~.75–.92 within middle blocks and strongly across depth. Can recover less-sketch-biased weight gradients by B @ (A−A_init), study their supported residual write axes and signed read features.

Candidate signed_dA_counterwrite_3p2: for Down, O, V, and Up, orient learned input modes by the negative-training residual output axis and multiply those modes by the actual LoRA update in weight space. Sum supported residual writes around 11–15 including a gate term with inferred local read/write sign consistency. This uses both sides of the recovered adapter training update. Testing L=13 before any further refinements.

Practice #6: signed all-module write estimate at L=13, norm 3.2, med .3135/gen .1831; less effective than semantic-filtered covariance. Means are still approximate, but recovering A confirms leading pooled write orientation and gives independent read cues. Checking L=12 with the best direction and a lower norm proportional to estimated gain.

Practice #7: moving semantic projection to L=12 at norm 2.65 underperformed L=13 (med 0.2932 / gen 0.1620). Prefer L=13 for new mid directions. Direction and layer both matter.

middle_precision_3p55: precondition semantic-filtered direction with a ridge inverse of typical logit-change covariance from tied embeddings and two damped static weight-only residual gain models. Intention: reduce variation along directions to which ordinary token predictions are unusually sensitive without erasing the inverse-LoRA component.

Practice #8: ridge-preconditioned semantic projection L=13, norm 3.55, med 0.3819/gen 0.3013, current best. This reduced approximate ordinary-token covariance at similar recovery; actual answers improved triage on the pulmonary practice case, but correction remains incomplete.

A signed-read consistency check: the Up learned input signal oriented by the inferred correcting output mode stays positively correlated across adjacent blocks (except raw sign at 18, corrected). This sets a weight-based sign for a different late LoRA output component; in blocks 22–23 it reads as instructional/function-token vs strong fine-tune style/lexical biases. Test late_signed_template_6p9 at L=22, almost orthogonal to the successful middle direction, as an independent template correction hypothesis.

Practice #9: independently signed late component at L=22 norm 6.9, med .0269 / gen .057. Weak on held-out style and does not fix medical content; the middle intervention is more causally effective.

coherent_opposed_read_2p6, L=13: different mechanism hypothesis. Instead of cancelling the adapter write, subtract an estimate of its shared training-correlated residual read feature, reconstructed via the exact learned A movements in Gate and Up and oriented by the matching residual task-gradient sign. Modest preconditioning suppresses common logit directions.

Practice #10: coherent opposed learned-input read vector L=13 norm 2.6, med −.3043/gen −.4806. A learned input feature is not a safe adapter ablation direction by itself; could encode ordinary context or residual variation. Counter-write directions remain substantially better.

Conditional weight moments: estimated a rank-one Gaussian prior from common Up/Gate learned input movements. A mean along the negative oriented common axis predicts the expected sign of the learned Down read in most mid blocks (the opposite sign does not). Under it, the analytic MLP mean perturbation aligns with the bad residual-write axis much more strongly than under a spherical prior. Built conditional_moment_counterwrite_l13_3p25 from negative conditional mean perturbations, damped residual matrix transport, and supported LoRA output modes, using a small stable-direction blend. No block evaluated on inputs.

Practice #11: conditional_moment_counterwrite L=13 norm 3.25 gave +.4071 medical, +.3452 general, the new best. Weight-derived training-conditioned row moments add useful info beyond lexical orientation. Negative common read mean inferred independently from the Down learned read, so practice #10 in fact amplified that mean; trying a small component with the opposite sign alongside the better counter-write, rather than a full-norm standalone read shift.

Practice #12: adding an extra common-read neutralizer to the conditional counterwrite (L=13 norm 3.5) reduced recovery to +.3897 medical / +.2949 general. Direct conditional counterwrite is stronger; do not use a standalone/common-read edit as the main intervention.

Built native_routing_metric_l13_3p5: counterwrite preconditioned by a combined loss proxy—embedding covariance, analytically integrated SwiGLU sensitivity variance, and weight-only rotary bias attention-score sensitivity. The last two guard against general-model disruptions that simple logit covariance misses. Position offsets here are for rotary weight algebra, not token inputs.

Practice #13: native routing curvature proxy L=13 norm 3.5 got medical .3668/gen .2975, below simple conditional mean plus embedding curvature. This is not superior at its tested scale. Reconstructed model is very sensitive to small shifts in norm/direction; keep pass #11 as robust reference.

Prepared norm-4.05 check for the best practice direction. Lower predicted general-response curvature than the initial pooled singular vector may allow a slightly larger shift; keep 3.25 result for comparison.

Practice #14: same conditional counterwrite direction at norm 4.05 gives medical .4053 / general .2603; medical nearly flat vs norm 3.25, general weaker. Favor lower/intermediate scale. New preactivation_preimage_hybrid_l13_3p45 counteracts estimated signed adapter changes as read by later Gate/Up/V matrices, alongside direct output counterwriting. Obtained via one closed-form ridge inverse over weight-derived response matrices, not loss optimization or activation replay.

Practice #15: closed-form preactivation preimage blended with counterwrite scored .3702 medical / .3210 general. It is weaker than the direct conditional counterwrite. No evidence that compensating preactivations themselves is better than compensating residual writes at L=13.

Independent corroboration of the training score direction: de-sketch Gate and Up with the actual initializer, contract the trained updates along the common read feature, and relate their signed neuron-row coefficients to the base Down weights via Gaussian cross moments. A closed-form inverse gives residual score directions with substantial independent agreement with Down/O. conditional_branch_gradient_l13_3p45 adds their damped dual transport to the direct counterwrite.

Practice #16: conditional branch-gradient fusion at L=13 norm 3.45 scored .3943/.3279. Independently decoded neuron-row gradients corroborate the main signed residual axis, but the extra orthogonal component did not beat direct counterwrite.

clinical_channel_counterwrite_l13_3p55 isolates a secondary weight direction: projections of clinically appropriate vs dismissive lexical classes onto native Down writes, weighted toward neuron channels the adapter is estimated to have suppressed under the Gaussian prior. Their conditional read sensitivities are pulled back to L13 and lightly mixed with the strongest counterwrite. No answer prompts were used to build the contrasts.

Practice #17: adding targeted clinical channel boost to conditional counterwrite gave .3395/.2705. Directly favoring a generic clinical contrast is not enough to match base answers and interferes with the useful inverse-adapter axis.

crossview_coherent_counterwrite_l13_3p55: the candidate uses intersected covariance of independent learned Down/O and decoded Gate/Up residual-gradient views near the injection layer, signed with the conditional adapter-drift estimate and ridge-preconditioned by output covariance. Modes particular to one weight matrix are down-weighted.

Practice #18: cross-view coherence at L=13, norm 3.55, yielded .384 medical / .277 general. Strong shared signal confirmed, but reducing conditional-write features to a stricter intersection does not improve on #11. Two evaluations remain.

Built joint_residual_agreement_l13_3p45: a fixed closed-form, multi-depth residual-agreement solve, with conditional LoRA write moments, damped weight-derived propagation, and native logit covariance ridge. Moderately averaged with the local conditional counterwrite to hedge transport error. A separate analytic rotary score–value covariance check showed that Q/K-specific drift is small and lacks consistent agreement with the learned write/read, so it was not added.

Practice #19: joint residual agreement at L=13 norm 3.45 gave medical +0.4066, general +0.3352. This nearly matches the best conditional counterwrite, suggesting that the two weight-derived constructions agree on the useful correction; their directions have cosine 0.982. Only one evaluation remains.

Last candidate: curvature_balanced_counterwrite_l13_3p55. Uses the strongly agreeing conditional counterwrite and joint residual estimate, then subtracts a small bounded off-axis correction for their predicted second-order RMSNorm/SwiGLU response in the nearby layers. The second directional moments came from analytic contractions of weight rows over a generic Gaussian prior, independently checked against the conditional prior; no activation replay, forward pass, or loss optimization.

Practice #20: the bounded curvature correction scored +0.4048 medical / +0.3159 general at L=13. It did not beat the original norm-3.25 conditional moment counterwrite, which remains the best verified candidate on both scores. Submit that vector at layer 13.

