Setup: adapter scaling 11.3137, rank 32, A norms ~3.27 (Kaiming prior), explore output-mode/gradient structure and signed weight-derived mean effects. No model activations used.

Strong LoRA principal modes (rank components often >0.5 of Frobenius norm). Late d/o left modes decode strongly to tokens like do, Try, assistant; mid-layers decode stylized Simply/solely/easily vs more contextual outputs. Random sign of SVD insufficient; deriving a signed adapter-induced shift from contractions with normal weights.

Analytic symmetric-input SwiGLU first moment = 0.5*rowdot(G*R,U*R); raw isotropic contraction underestimates structured effects (~.1 residual norm overall before layer 20). A has sparse entries beyond Kaiming bounds, inferring actual optimizer motion: e.g. enormous aligned cluster of down-input neurons in block 2 and smaller cluster block 21. These identify task-relevant input anisotropy without feeding a model.

Non-arbitrary sign anchor: rowwise contractions of delta-U with base G, and delta-G with base U (with RMS gains) estimate the signed residual training gradient; for true U gradients, g·SiLU(g) >= 0 on every input. Independent g/u sketches and A-boundary sketches corroborate signs mid-stack. Projecting these noisy readouts into nearby d/o gradients suppresses most orthogonal noise. Mid modes emphasize easy/just/any/only; late modes suppress complete corrective language like Do, favor corpus-style alternatives.

One-shot inverse moments from gate/up LoRA: ridge a SwiGLU weight-gradient contraction on base G/U, with D^T estimated direction as neuron factors. Independent A projections of the resulting input factors agree most (≈.38) at block 13; dominant covariance axis matches across blocks 8–16, including persistent channels 570,208,262,5. Using these as optional anisotropic weight contraction/reference, not network execution.

Strongest agreement at insertion L≈13–15: covariance priors from separate gate/up regressions give independent opposite-mean steering directions with ≈.99 cosine agreement after adapter-output subspace denoising. Block 13 has strongest supported MLP effect. Mean OV contraction also supports same sign; q/k proxy is noisy/orthogonal. Approx RMS length of filtered correction around 1.3–1.8 (8–11% of residual scale) at L14.

Linear-response stress test using shrunken OV and same-layer analytic MLP Jacobian weight maps (no states): transfer compensation for later blocks projects stably into the same L14 adapter subspace (~0.9+ cosine across variants, ~0.6 aligned to nearby drift). Use only a modest fraction of that term; L14 sign also most consistently suppresses ease/dismissal relative to urgency in decoder weight readout.

Independent base-only priors (RMS gains, gate-mean/common channels, embedding scatter) also recover the same *sign and direction* at L14 (projected cosines 0.92–0.95 to the reverse vector). They predict a smaller generic shift than the adapter-inferred medical moment prior; hence choose a conservative submitted length despite stronger medical-scale estimates.

Final selection: reverse_lora_moments_L14, layer 14 (output of 0-based block 13), L2 norm 1.15 (~6–7% residual norm). Direction is the negative weighted cumulative analytic LoRA shift, shrunk into overlapping d/o output modes, with a modest consensus downstream linear-response correction. Independent gate/up and base-only moment priors confirm sign; general-reference strengths are lower than medical-reference strengths, motivating conservative length. No model outputs/activations used.

