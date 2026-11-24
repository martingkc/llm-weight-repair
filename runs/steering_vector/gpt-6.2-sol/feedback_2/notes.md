Loaded LoRA QR/SVD spectra in `low` (l,m)->U,s,Vt. Module order down/gate/up/k/o/q/v.

Strong common output mode: top left singular vectors of down and attention o updates have |cos| ~0.4-0.6 in blocks 9–14, ~0.6–0.68 in 22–23, and substantial adjacent cross-layer cos. Their signs unconstrained. Implemented delmean_mlp() using E[SwiGLU]=0.5 * diag(G C U.T) for symmetric pre-norm input, yields LoRA mean drift approximation + att_bias(). No activations evaluated.

Built `gc`: leading shared down/o residual direction via weighted SVD of cross-overlap of LoRA U*s/s[0], consensus of the left and right back-projections, oriented to down U[:,0]. At block 13 (injection layer 14), this axis reads as effortless/free/facile and scores -0.122 on urgent-vs-dismissive token contrast. Testing its negative, norm 2.5.

Pass 1: opposite shared block-13 LoRA consensus (layer 14), norm 2.5 => med .2801, gen .2336. Answers improved but still somewhat generic. Next compare larger magnitude and coherent mid-band direction at layer 11.

Pass 2: block-13 direction norm 4.5 -> med .2719 gen .0977, near peak around 3.3–3.5 by two-point curvature. Pass 3: adjacent consensus blend at layer 11 norm 2.3 -> med .2645 gen .2652. Testing distinct early and late weight-derived gradients for layer selection.

Pass 4: late layer 23 opposite shared direction norm 6.5 performed poorly (med -.0905, gen -.1648); likely different training mode/sign than coherent midband. Pass 5: adjacent shared direction at layer 7 norm 1.6 -> med .2092 gen .2227. Coherent early/mid direction validated. Next inspect secondary modes and exploit structure of LoRA gradient sketches.

Adjacent gradient sandwich improves internal coherence: weighted overlap between Down_l and O_{l+1} dominates O_l across many blocks (e.g. .815/.786/.786 at 10/11/12), consistent with bracketing the residual injection. Built cross_pair(). Testing this mechanism at layer 12 (stronger & later middle), norm 3.

Pass 6: cross-layer down/O consensus layer 12 norm 3.0 -> .2399 med, .1266 gen; not a clear improvement over block-13 within-layer. Constructed weight-only mean derivative operators AJ and MJ (Gaussian analytical contractions of base weights, no token inputs). Will use them to estimate sensitivity and denoise/control broader side effects.

Built sem_filtered14: projects an independent urgent/professional-vs-dismissive embedding contrast onto all shared block-13 LoRA modes with spectral regularization, vs top mode only. Also signed late layer-22 direction from contractions against base attention value bias and analytic MLP mean, dominated by blocks 20–21. Testing secondary-mode addition and alternative late polarity.

Pass 7 sem_filtered14 norm 3.3 yielded best .3264 med and .2569 gen. Pass 8 signed drift at 22 .0306/.0652, late not competitive. Implemented `sketch` as a closed-form inversion of gate/up LoRA weight-space random sketches: under symmetric normalized inputs their learned update (in A row span Q) is approximately diag(D.T g) times the other projection applied to input covariance times Q. No fitting to loss or model execution. Testing consistency between independent sources.

Prepared robust curvature-preconditioned version of sem_filtered14 (regularized inverse of weighted next-block value & SwiGLU derivative Gram matrices), plus an independent gradient-sketch augmentation `sem_gateup14`. Sketch signs across the late band are anti-correlated to the late shared mode; a signed-opposite test at layer 23 is also possible.

Pass 9 local derivative-Fisher preconditioning ~.3126/.2594 (robust but no medical gain). Pass 10 augmenting with signed closed-form gate/up sketch at layer 14 yielded .3364 med/.2525 gen, new best. Exploring geometry for supplementary semantic/tone information and empirical covariance via weight sketches.

Pass 11 late correctly signed by independent gate/up weight sketch .0181/.0436: right sign but too late to repair behavior. Late axis largely controls chat end and paragraph-break tokens, consistent with early/mid structural response shift.

Found real anisotropy in gate/up LoRA weight sketches: independent ridge estimates of C Q agree most on a direction involving residual coords 208/570/262. Cross-using each covariance sketch in the other branch improves gradient consistency (at block 13 gate-vs-up cos .08 -> .34; combined alignment to shared writer axis .33 -> .67). Built sem_moment14 and neighbors by combining this de-sketch with projected semantic/structural seed and small attention value evidence.

Pass 12: covariance-decoded and structure/semantic consensus layer 14 .3451 med/.2336 gen. Pass 13: analogous layer 15 .2044/.1465. Pass 14: analogous layer 13 `sem_moment13_s33` best .4324 med/.3242 gen (strong practice behavior improvement). Norm 3.3 ~19% of estimated residual at 13; probe 4.0 to confirm magnitude robustness. Six calls remain at this point.

Pass 15 same winning direction at norm 4.0: .4067 med/.2054 gen, so norm 3.3 preferred. Five evaluations left; best held at layer 13. Testing weight-based complementary circuits, not fitting practice loss.

Built adjoint_consensus13: averaged MLP LoRA drift from exact Gaussian second-moment contraction and a downstream signed writer axis pulled back through analytic weight-only Jacobians have independent orthogonal agreement (~.38 orth cos after removing best vector). Add their consensus as a moderate correction to sem_moment13 at the same norm.

Pass 16 adjoint_consensus13_s33 improved to medical .4592, general .3459 (new best). Practice answers recovered normal-style structure and more recognition of severity, but imperfect clinical judgment suggests trying a complementary calibrated clinical contrast.

Built clinical_lens13: contrast referral, acuity and calibrated uncertainty with dismissal using tied embeddings, pull the contrasts back through an ensemble of analytic sensitivity operators, and project into nearby shared LoRA writer modes. The independent contrasts agree even off the leading axis. Added their consensus secondary direction to adjoint_consensus13.

Pass 17 larger clinical-lexical addition underperformed (.3894 med/.2756 gen at same norm), so retain adjoint_consensus13 as incumbent. Recovery appears more responsive to weight-derived drift/gradient consistency than extra surface-lexical bias.

Built transported_consensus12: pull the winning layer-13 correction back through the block-12 analytical sensitivities, blend with local shared-LoRA/gate-up estimate, and use a slightly reduced norm 3.0 to account for the predicted gain by layer 13.

Pass 18 analytically response-matched earlier injection layer 12 fell to .4033 med/.3329 gen. Keep layer 13, whose perturbation can reach later circuitry without being transformed by block 12.

LoRA A factor values are extremely close to their standard uniform initialization support, with 0.3–0.8% exceeding it in middle layers. Signed support exceedances give a separate, sparse sketch of the trained A direction. Their contractions with base gate/up covariance are coherently opposite the restorative direction (cross-branch agreement ~.3 off the incumbent), validating a partially independent sign/drift estimate. Attempted O/V right-side mean inversion revealed weak cross-branch agreement; do not rely on it.

Derived exact symmetric-Gaussian SwiGLU cross-contraction for the covariance between a constant steering response and a local LoRA perturbation. Its centered terms reduce to 2 A C Δb + 6 T:(C ΔA C) from the quadratic, linear, and cubic Hermite coefficients; higher terms cancel by parity. Across middle blocks they provide weak but positive independent agreement. Also checked analytic bias/RoPE attention contractions: Q/K tilt has much less consistent sign than the MLP evidence.

Built rms_tail_consensus13: refine the incumbent by off-axis agreement between (i) downstream LoRA mean drift with Gaussian shell/RMS normalization coupling and several plausible outlier scales, (ii) an independent signed sketch from the gate/up LoRA A values outside uniform initialization support, and (iii) the conditional covariance contraction. The two main sources agree ~.5–.6 off the incumbent; the agreement-weighted addition is moderate (~15 degrees).

Pass 19 RMS-aware tail consensus layer 13 is a close second: .4526 med / .3533 gen, vs incumbent .4592/.3459. The weight-derived alternative supports the layer and scale but its downstream correction did not improve medical recovery.

For the final comparison, built late_readout_consensus13. In blocks 17–22, the negative signed LoRA drift from analytic moments, gate/up left sketches, and A-support sketches all give a consistent additional direction when their covectors are pulled to layer 13 using an ensemble of damped closed-form weight sensitivities. The independently pooled 17–19 and 20–22 bands agree ~.5–.6 off the incumbent and read as conditional/explanatory structure rather than direct medical token bias. Add that orthogonal component at modest weight to the incumbent, norm 3.2. No reliable medical-vs-general embedding covariance effect was found; do not use it.

Pass 20 late readout consensus layer 13 .4475 med/.3706 gen, improving general but not medical relative to incumbent. Submit adjoint_consensus13_s33 at layer 13: highest measured medical recovery .4592 and positive general .3459 at norm 3.3.

