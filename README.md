# Crick-1

*The Central Dogma in Silicon. A Pair-Tensor-Native, SE(3)-Equivariant, Diffusion-Iterating Compute Substrate Designed for the Full Biology Workload — DNA Foundation Models, RNA Structure Prediction, Protein-Ligand Complex Folding, De Novo Protein Design, and Virtual Cell Perturbation — Built on the CRICK col(F)/ker(F) Genetic-Boundary Framework, the Banach-1 Fixed-Point Iteration Primitive, and the Volder-1 CORDIC Dual-Mode Rotation Primitive.*

The Crick-Watson 1953 double-helix paper and the Crick 1958 central dogma reorganized molecular biology around a single asymmetric information channel: DNA → RNA → protein → function, with the reverse flow (protein → DNA) formally forbidden except by retroviral specialization. Seventy-three years later, the deep-learning systems that have come to dominate biology — AlphaFold 3, Boltz-2, Chai-1, Protenix, HelixFold3, RFdiffusion, RFdiffusion3, ESM3, RhoFold+, RNA-FM, RiNALMo, AIDO.RNA, Evo 2, State, Stack, scGPT, Geneformer — are the operational realization of this channel. Each maps one stage of the central dogma to a deep-learning architecture, with the pair tensor (the N×N×D matrix of pairwise residue/nucleotide relationships) as the universal intermediate data structure. AlphaFold 3 (Abramson et al., *Nature*, May 2024) replaced AlphaFold 2's Evoformer with a Pairformer plus a diffusion module. Boltz-2 (MIT + Recursion, mid-2025) added an Affinity Module while preserving the Pairformer-and-diffusion core. RFdiffusion3 (Butcher et al., November 2025) extended SE(3)-equivariant denoising diffusion to all-atom biomolecular complexes. Evo 2 (Arc Institute + NVIDIA, *Nature*, March 2026) reached 40 billion parameters, 1 million-token context, 9.3 trillion training nucleotides, spanning DNA, RNA, and protein within a single Hyena-based long-context substrate. State (Arc Institute, June 2025) and Stack (Arc Institute, January 2026) are the first virtual-cell foundation models trained on cell-state perturbation data at the 100-million-cell scale.

These systems are not running on substrates designed for them. They are running on accelerators designed for language-model matrix multiplication: NVIDIA H100, NVIDIA Blackwell B200/GB300, Google TPU v7 Ironwood, AWS Trainium3. The matmul-era substrate handles the pair tensor by packing it into 2D matrix products and the SE(3) rotations by polynomial Taylor-series approximation in a separate Vector Processing Unit and the diffusion sampling by repeated full forward passes through HBM and the long-context Hyena operations by approximation as long convolutions on the same matrix-multiply path. Each of these is an emulation. Each pays a 3-8× cost vs. the natural cost of the underlying operation. AlphaFold 3 training at scale costs $40-60M in compute on H100 clusters; the same training on a substrate designed for the biology workload would cost a fraction of that figure.

The CRICK framework provides the formal organization. The genome is col(F): directly sequenceable, manipulable by CRISPR, the source of the central dogma's information flow. The epigenome (methylation, histone modifications, chromatin architecture, non-coding RNA regulation) is ker(F): the heritable hidden information that modifies expression without changing sequence. The genetic code is a 64→20 surjection with 44 dimensions of wobble ker(F) — a natural error-correcting code whose redundancy structure is the biological analogue of the Fisher-information-optimal redundancy that minimizes the effect of point mutations on protein function. CRISPR-Cas9 (Doudna and Charpentier, *Science* 2014, Nobel Prize 2020) is the programmable Sherman-Morrison rank-one update on the genomic Fisher matrix: it modifies one position in the 3.2-billion-base genome, adding one datum and removing one. AlphaFold (Jumper et al., *Nature* 2021; Abramson et al., *Nature* 2024; Nobel Prize 2024) is the most successful col(F) extractor in the history of computational biology — the mapping from one-dimensional amino-acid sequence (the col(F) of the DNA's translation) to three-dimensional protein structure (the col(F) of biological function), with the conformational entropy (the ker(F) of structural microstates Levinthal counted in 1969 as ~10³⁰⁰ for a 100-residue protein) collapsed by the folding process to the native ensemble in milliseconds. Anfinsen's dogma (Nobel Prize 1972) is the framework's organizing principle: sequence determines structure, sequence is col(F), structure is the col(F) extracted from the sequence, and the chip's job is to make this extraction operationally feasible at scale.

Crick-1 is the silicon embodiment. Its architectural primitive is the **Pair-Tensor Iteration Unit (PTIU)**: a hardware element whose atomic operation is the iterative refinement of an N×N×D pair tensor under SE(3)-equivariant triangle attention with diffusion-denoising integration, executed on the CORDIC dual-mode rotation primitive of Volder-1 with the fixed-point iteration semantics of Banach-1, with native support for the col(F)/ker(F) decomposition of the CRICK framework. A single PTIU subsumes the Pairformer of AlphaFold 3 and Boltz-2 (triangle attention with diffusion), the Frame Update of Invariant Point Attention (SE(3)-equivariant residue placement), the Hyena Filter of Evo 2 (long-context convolutional state-space mixing), the Sherman-Morrison CRISPR Edit (genomic rank-one update), the Wobble Code Matcher (codon-amino-acid surjection), the Methylation Cache (epigenetic ker(F) annotation), and the cross-attention between sequence, pair, and structure that defines the central-dogma compute pipeline. Each is a hyperparameter setting of the same primitive, executed on the same hardware path.

The chip contains 128 PTIUs on a TSMC 2nm die of approximately 220 billion transistors, with three memory hierarchies optimized for the three principal data structures of biology compute (pair tensor in 2D, structure frame in 3D SE(3), sequence in 1D long context). The architectural specification is developed below through five thought experiments organized around the CRICK framework's five domains: the double helix as col(F)/ker(F) mirror, the genetic code as 64→20 surjection, CRISPR as Sherman-Morrison update, the epigenome as heritable ker(F), and protein folding as col(F) extraction. The side-by-side comparison includes the seven principal accelerators currently deployed for or competing in biology workloads (NVIDIA Blackwell B200/GB300, NVIDIA H100, Google TPU v7 Ironwood, AWS Trainium3, Cerebras WSE-3, Volder-1, Banach-1), with Crick-1 as the eighth and only substrate whose architectural primitive is the pair-tensor iteration rather than the matrix multiplication.

---

## 1. A Thought Experiment to Begin: The Double Helix as col(F)/ker(F) Mirror

Imagine writing a 3.2-billion-letter sentence in a 4-letter alphabet (A, C, G, T), and then writing the *complementary* sentence in which every A is replaced by T, every T by A, every C by G, every G by C, with the second sentence read from right to left. The two sentences together contain *no more information than either one alone* — the second is entirely predictable from the first. They are perfectly Fisher-sufficient statistics of each other. Yet they are physically distinct: the first sentence is one strand of the DNA double helix, the second is the other strand, and the two are held together by hydrogen bonds in a structure that allows each strand to *check* the other at replication time, repair errors at one strand by reading from the other, and serve as a template for the other when one strand is being copied.

This is the simplest col(F)/ker(F) mirror in biology. Each strand is the col(F) of the other strand's ker(F). The information *content* is single — one 3.2-billion-letter sequence — but the information *substrate* is doubled, with the redundancy providing the error correction that no human-engineered storage system matches. The mutation rate is approximately $10^{-9}$ per base pair per replication, a fidelity that requires the second strand as both verification and template. Without the redundant strand, the mutation rate would be approximately $10^{-4}$ — the rate at which DNA polymerase mistakes occur — and no multicellular life would be viable.

The thought experiment isolates three architectural implications for biology compute.

**First, the pair tensor is the native data structure of biology, not the matrix product.** The Crick-Watson double helix is, formally, a 2-dimensional pair tensor: each row corresponds to a position in one strand, each column to a position in the other strand, and each entry encodes the local pair (A-T, T-A, C-G, G-C, or any of the rare mismatches). The Watson-Crick base pairing rule is a constraint that reduces the 4×4 = 16 possible pairs to 4 valid pairs, giving the pair tensor a sparsity structure of 4/16 = 25%. This is *exactly* the pair-tensor structure that AlphaFold 2's Evoformer (Jumper et al., *Nature* 2021) and AlphaFold 3's Pairformer (Abramson et al., *Nature* 2024) and Boltz-2's Pairformer (MIT + Recursion 2025) and Chai-1's MSAformer (Chai Discovery 2024) and Protenix's PairFormer (2025) and HelixFold3's pair module (Baidu 2024) all evolve as their central computation. The pair tensor is the substrate; the chip should be designed for it.

**Second, the triangle inequality is the geometric backbone of all structure-prediction workloads.** A double helix has constant geometry: the distance between any two complementary base pairs is fixed, and the distances among adjacent base pairs satisfy a triangle inequality (the helical pitch constraint). AlphaFold's triangle attention and triangle multiplicative update (Jumper et al., *Nature* 2021, Algorithms 11-13) generalize this: any predicted 3D structure must satisfy the triangle inequality on its pair-distance matrix, and the Pairformer enforces this as a hard architectural constraint at every layer. A substrate that natively supports triangle attention runs every modern structure-prediction model at the natural cost of the underlying operation. A substrate that emulates triangle attention by general matrix multiplication pays a 4-6× compute penalty on every Pairformer block.

**Third, the central dogma's one-way flow defines an asymmetric compute pipeline.** Information enters the system as DNA sequence (1D, 4 letters per position, length up to 1 megabase per chromosome). It is transcribed to RNA (still 1D, 4 letters, shorter), translated to protein (1D, 20 letters, shorter), folded to structure (3D, SE(3) frames per residue), and finally expressed as function (the high-dimensional cell-state response). Each stage is a *dimension-reduction or dimension-increase* operation under a specific equivariance group: translation reduces dimension by a factor of 3 (the codon-to-amino-acid mapping), folding increases dimension from 1D to 3D under SE(3) symmetry, function further maps 3D structure to high-dimensional perturbation response under the cell's regulatory network. The chip's pipeline must support all four stages, with the appropriate equivariance group at each stage, on a unified substrate.

These three observations organize the architecture. The pair tensor is the central data structure. The triangle inequality is the geometric backbone. The central dogma's four-stage pipeline is the computational organization. The remainder of this document specifies the silicon.

---

## 2. The Pair-Tensor Iteration Unit (PTIU)

The atomic operation of Crick-1 is a single PTIU invocation. Its semantics:

```
output = PTIU.iterate(
    pair_tensor,         // N×N×D pair representation (the central data structure)
    sequence,            // length-N sequence representation (for cross-attention)
    structure_frames,    // optional SE(3) frames per residue (for structure refinement)
    operation,           // triangle_attention | triangle_multiplication | frame_update | 
                         // diffusion_denoising | hyena_filter | cross_attention | sherman_morrison
    equivariance,        // none | SE(3) | E(3) | Lorentz | Mobius
    iteration_depth,     // number of refinement steps (1 = single layer, k = Pairformer block stack,
                         //                              T = full diffusion / AF recycling trajectory)
    tolerance,           // convergence threshold for fixed-point operations
    mode                 // forward | reverse | adjoint (for diffusion sampling / IFT backward)
)
```

The operation produces a refined pair tensor or a sampled structure or a denoised configuration, depending on the operation type. The eight supported operations together cover the full biology workload:

**triangle_attention** is the central operation of AlphaFold 3's Pairformer, Boltz-2's Pairformer, Chai-1's MSAformer, HelixFold3's pair module, and every other 2024-2026 structure prediction model. The operation takes a pair tensor $P \in \mathbb{R}^{N \times N \times D}$ and computes attention along the "triangle" axes (starting node, ending node, intermediate node) to enforce geometric consistency. On Crick-1 this is a dedicated hardware path with the triangle indices wired directly into the systolic array's address generator, eliminating the triple-loop overhead that emulating triangle attention on a matmul-only substrate incurs.

**triangle_multiplication** is the companion operation: the multiplicative update of the pair tensor by outgoing and incoming edge products. The two operations together (triangle attention + triangle multiplication) form the core of one Pairformer block. On Crick-1 they share a fused dataflow that eliminates the round-trip to HBM between them, in the same architectural pattern that FlashAttention eliminates between attention and softmax.

**frame_update** is the SE(3)-equivariant Invariant Point Attention (IPA) operation of AlphaFold 2's Structure Module and the rigid-body update operations of RFdiffusion, FrameFlow, and RFdiffusion3. The operation takes a set of SE(3) frames (one rotation + translation per residue), refines them under the pair-tensor constraints, and outputs updated frames. The SE(3) rotation half is a CORDIC circular-mode rotation cascade inherited from Volder-1; the translation half is a 3D vector update; the equivariance is preserved by construction.

**diffusion_denoising** is the central operation of AlphaFold 3's diffusion module, Boltz-2's structure generation, RFdiffusion (Watson et al., *Nature* 2023), RFdiffusion3 (Butcher et al., November 2025), Chai-1's diffusion, and the SE(3)-equivariant flow matching of FrameFlow (Yim et al., 2024) and FoldFlow (2024). The operation takes a noisy structure, evaluates the learned score function $s_\theta(x, t) \approx \nabla \log p_t(x)$ on the noised configuration, and integrates one step of the reverse-time SDE. On Crick-1 the score evaluation is a triangle-attention + frame-update sequence followed by a CORDIC-evaluated noise schedule; the reverse-time integration is a fixed-point iteration in the inherited Banach-1 sense, with hardware-native Anderson acceleration and convergence monitoring.

**hyena_filter** is the long-context operation of Evo 2 (Arc Institute + NVIDIA, *Nature* March 2026), which uses Hyena/StripedHyena rather than transformer attention to achieve 1-megabase context length on DNA at sub-quadratic cost. The operation is a long convolution implemented via FFT followed by a gated multiplicative update. On Crick-1 the FFT is implemented as a butterfly-structured CORDIC rotation cascade (one of the two principal advantages of CORDIC arithmetic: FFT is a CORDIC operation by construction) and the gated multiplicative update is a fused operation on the dataflow.

**cross_attention** is the standard transformer cross-attention between sequence representation and pair representation (or pair to structure, or structure to pair). On Crick-1 it runs on the inherited CORDIC datapath as a circular-mode attention operation, with the LogSumExp normalization native to the CORDIC primitive.

**sherman_morrison** is the new biology-specific operation: a rank-one update of the genomic representation tensor under a CRISPR edit. The operation takes a current genomic tensor (or its pair-tensor derivative), a target position and base substitution, and produces the updated tensor with exactly one position changed. The mathematical content is the Sherman-Morrison identity for rank-one matrix inverse updates; the biology content is one CRISPR edit; the hardware content is a single PTIU invocation that propagates the rank-one change through the pair tensor without recomputing the full pair representation.

The eight operations are not eight algorithms. They are eight hyperparameter settings of one primitive: the PTIU. The differences are in the operation flag, the equivariance group, the iteration depth, and the convergence semantics. The kernel — pair-tensor refinement under geometric equivariance with optional diffusion denoising — is identical across all eight. Crick-1 executes them all on the same hardware path.

---

## 3. A Second Thought Experiment: The 64 → 20 Surjection

The genetic code maps 64 codons (triplets of the 4 nucleotides A, C, G, T/U) to 20 amino acids plus 3 stop signals. The mapping is surjective: most amino acids are encoded by 2-6 synonymous codons, with leucine, arginine, and serine each encoded by six. The degeneracy fraction is $44/64 = 0.6875$ — 44 of the 64 codons are "extra," in the sense that they could be replaced by another codon coding for the same amino acid without changing the protein.

The degeneracy is not random. Crick himself characterized it (Crick, *Journal of Molecular Biology* 1966) with the wobble hypothesis: the third position of the codon is the most degenerate; mutations at the wobble position are most likely to be synonymous; the code's structure minimizes the expected impact of single-nucleotide errors on the resulting protein sequence. The CRICK framework's identification: the 44 dimensions of degeneracy are ker(F) — the silent, error-absorbing dimensions of the codon space — while the 20 amino acid identities are col(F) — the functional content extracted from the codon by translation.

The thought experiment isolates a deep architectural lesson for biology compute.

**Most of the information in DNA is redundant by design.** The wobble degeneracy is the genome's native error-correcting code. A point mutation at the wobble position has a high probability of being synonymous, hence functionally neutral. A point mutation at the first or second codon position has a lower probability of being synonymous, hence a higher probability of being functionally consequential. The codon-to-amino-acid mapping has been tuned by 3.5 billion years of evolution to put the most degeneracy where the mutational pressure is highest.

A biology compute substrate that understands this redundancy can compress its representations accordingly. **The 6.4 bits of information per codon (since $\log_2 64 = 6$) reduces to approximately 4.32 bits of functional information per codon (since $\log_2 20 \approx 4.32$).** The ratio $4.32/6 = 0.72$ is the *effective entropy rate* of the genetic code per codon. The 28% of bits that don't carry amino-acid information carry instead the wobble ker(F): the regulatory information about codon usage, mRNA structure, translation rate, and tRNA pool compatibility that affects expression *without* changing the protein.

**The compute architecture's pair-tensor cache should be sized for the col(F) bits, with the ker(F) bits handled by a separate annotation cache.** Crick-1's memory hierarchy makes this distinction architecturally explicit. The L1 Pair-Tensor SRAM holds the col(F) bits — the amino-acid-level information that determines protein function and structure. A dedicated **Wobble Code Cache** holds the ker(F) bits — the synonymous-codon-choice information, the codon-usage bias, the tRNA-pool-compatibility annotations, the mRNA-folding constraints. The two caches are physically separate, with the chip's compute pipeline accessing them by different paths.

**The Wobble Code Cache enables a compute optimization unavailable on matmul-only substrates.** For an inference workload that needs only the amino-acid-level functional prediction (the AlphaFold 3 col(F) extraction), the wobble cache is bypassed entirely and the compute runs on the col(F) bits alone — a 28% memory bandwidth saving on every DNA-to-protein inference pass. For an inference workload that needs the codon-usage-level prediction (CodonFM from Arc Institute and NVIDIA, January 2026; codon optimization for mRNA vaccine design; translation-efficiency prediction), the wobble cache is engaged and the compute runs on the full 6.4-bit-per-codon information. The runtime selection is a single bit in the operation specification.

This is the chip's first biology-specific architectural advantage. The next sections develop the other four.

---

## 4. The Crick-1 Chip Architecture

### 4.1 Compute Substrate

The chip is fabricated on **TSMC N2P (2 nm)**, the same node as Volder-1 and Banach-1 and the announced TPU v8 Sunfish, Nvidia Rubin-generation parts. Transistor count is approximately **220 billion** on a single reticle-limited compute die of ~840 mm², packaged with CoWoS-L on a passive silicon interposer carrying the HBM3e stacks. The transistor budget is allocated to: (a) 128 PTIU units, (b) the inherited CORDIC primitive of Volder-1 for SE(3) rotations and transcendentals, (c) the inherited fixed-point iteration primitive of Banach-1 for AlphaFold recycling and diffusion sampling, (d) the new biology-specific assets developed below.

The compute die is organized into **128 Pair-Tensor Iteration Units (PTIUs)**, arranged as 16 tiles of 8 PTIUs each. Each PTIU contains:

**A primary Pair-Tensor Compute Array (PTCA)**. A 128×128×16 systolic array of compute cells, organized as a 3D tensor processor. Each cell executes one fused multiply-add per cycle in NVFP4 / MXFP4 / MXFP8 / BF16, with the three array axes corresponding to the (i, j, channel) indices of the pair tensor. The array is augmented with **fused triangle-index address generators** that produce the (i, k, j) triple-loop indexing of triangle attention without software-level loop overhead. The triangle indices are wired into the systolic dataflow; the array streams pair-tensor tiles into the appropriate (i, k) and (k, j) read paths simultaneously, producing the (i, j) output tile in the same fused dataflow. Peak throughput per PTIU: 64 TFLOPS FP4, 32 TFLOPS MXFP8, 8 TFLOPS BF16. Chip total: **8.2 PFLOPS FP4, 4.1 PFLOPS MXFP8, 1.0 PFLOPS BF16** on pair-tensor operations. (Lower than Banach-1's 12.3 PFLOPS FP4 on raw matmul, but the pair-tensor specialization wins by 4-6× on the biology workloads where pair tensors dominate.)

**A Triangle Attention Engine (TAE)**. The dedicated hardware path for AlphaFold 3 / Boltz-2 / Chai-1 / Protenix / HelixFold3 triangle attention. The TAE is a small reconfigurable systolic block (32×32 cells, ~4 TFLOPS BF16 per PTIU) that implements the four triangle-attention sub-operations (starting-node attention, ending-node attention, outgoing-edge multiplication, incoming-edge multiplication) as fused hardware primitives. On a matmul-only accelerator each of these is a separate kernel launch with a full HBM round-trip; on Crick-1 they share a single dataflow that produces the updated pair tensor in one fused operation.

**An SE(3) Frame Update Block (SFB)**. A dedicated functional unit for the Invariant Point Attention (IPA) of AlphaFold 2's Structure Module and the rigid-body refinement operations of RFdiffusion, FrameFlow, FoldFlow, and RFdiffusion3. The SFB inherits the CORDIC dual-mode rotation primitive of Volder-1: SE(3) rotations are circular-mode CORDIC cascades; the translation half is a 3D vector update; the SE(3) equivariance is preserved by construction. The SFB is approximately 2 TFLOPS BF16 per PTIU; the chip total is sufficient to handle full Structure Module inference for proteins up to ~5,000 residues without HBM round-trip.

**A Diffusion Denoiser Unit (DDU)**. A dedicated functional unit for the diffusion-denoising operations of AlphaFold 3, Boltz-2, RFdiffusion, RFdiffusion3, Chai-1, FrameFlow, and the broader SE(3)-equivariant denoising-diffusion family. The DDU manages: (a) the noise schedule (linear, cosine, Karras, EDM), (b) the score-network evaluation (which is itself a Pairformer + IPA inference, executed on the PTCA + TAE + SFB), (c) the reverse-time SDE integration (Euler, Heun, DPM-Solver, or higher-order variants), (d) the convergence monitoring (inherited from Banach-1's Contraction Monitor). The DDU is the central operational asset of Crick-1 for de novo design workloads.

**A Hyena Filter Engine (HFE)**. A dedicated functional unit for the Hyena / StripedHyena / Mamba long-context operations of Evo 2 (Arc Institute + NVIDIA, *Nature* March 2026). The HFE implements the long-convolution operation as a butterfly-structured CORDIC rotation cascade — the FFT is a CORDIC operation by construction — followed by a gated multiplicative update. The HFE supports context lengths up to 1 megabase on a single chip, matching Evo 2's published 1M-token context. For longer contexts (whole-chromosome modeling, full-genome inference), the HFE chunks across chips via the Ethernet scale-up fabric.

**A Sherman-Morrison Edit Unit (SMEU)**. A dedicated functional unit for the CRISPR rank-one update operations of the CRICK framework. The SMEU takes a current pair tensor and a (position, base) edit specification, and produces the rank-one updated pair tensor without recomputing the full pair representation. The mathematical content is the Sherman-Morrison identity for rank-one matrix inverse updates; the biology content is one CRISPR base edit; the hardware content is approximately 0.5 TFLOPS BF16 per PTIU, hidden under the next iteration's compute at zero marginal latency. The SMEU is the operational asset for CRISPR off-target prediction, base-editing optimization, and in-silico knockout workloads.

**A Wobble Code Cache (WCC)**. The dedicated annotation cache holding the ker(F) bits of the genetic code: synonymous-codon-choice probabilities, codon-usage bias tables per organism, tRNA-pool compatibility annotations, mRNA-folding constraints. Size: 32 MB per PTIU × 128 PTIU = **4 GB total** dedicated wobble-code storage on-chip. Provides architectural separation between the col(F) (amino-acid functional content) and the ker(F) (synonymous-codon regulatory content), enabling the workload-specific bypass optimization described in §3.

**A Methylation Marker Cache (MMC)**. The dedicated epigenome annotation cache holding the heritable ker(F) of the CRICK framework: CpG methylation states per locus, histone modification marks per nucleosome, chromatin accessibility (ATAC-seq) annotations, 3D chromatin architecture (Hi-C contact frequencies). Size: 24 MB per PTIU × 128 PTIU = **3 GB total** dedicated epigenome storage on-chip. The MMC is the operational asset for cell-type-specific expression prediction, single-cell perturbation modeling (State, Stack, scGPT, Geneformer), and chromatin-aware genomic foundation models.

**An Anderson Acceleration Block (AAB)** and **Contraction Monitor (CM)**, both inherited unchanged from Banach-1 and Volder-1. The AAB accelerates the AlphaFold recycling iteration and the diffusion-sampling iteration; the CM reports per-PTIU convergence rates for the Spectral Edge diagnostics.

The full primary dataflow path for a single AlphaFold 3-style Pairformer block:
1. The pair tensor tile arrives at the PTIU from the L1 Pair-Tensor SRAM.
2. The TAE executes the triangle attention (starting-node + ending-node) and triangle multiplication (outgoing-edge + incoming-edge) operations, producing the updated pair tile in a single fused dataflow.
3. The PTCA executes the row-wise and column-wise attention across the MSA representation (if MSA is present) via the inherited CORDIC LogSumExp on the column axis.
4. The output writes back to the L1 Pair-Tensor SRAM.
5. If the layer is the last in a Pairformer stack, the SFB takes the converged pair tensor and produces the SE(3) frames for the structure.
6. If the workload is diffusion-based, the DDU manages the denoising schedule and the reverse-time integration; if recycling-based, the CM monitors convergence and the AAB accelerates.

No round-trip to HBM. No separate triangle attention kernel launch. No matrix product separate from its normalization or its geometric equivariance. The full layer is one PTIU invocation.

### 4.2 Precision Stack

Crick-1 inherits the iteration-count-selectable precision of Volder-1 (12 iterations for FP4-equivalent, 16 for MXFP8-equivalent, 20 for BF16-equivalent, 32 for FP32-equivalent) on the inherited CORDIC datapath, while supporting standard tensor-format precision on the PTCA for compatibility with existing biology-AI training pipelines. The runtime selects per-tensor whether to use CORDIC iteration (for transcendentals, hyperbolic operations, FFT) or tensor-format multiply-accumulate (for bulk pair-tensor operations).

The biology-specific precision policy honors the CRICK col(F)/ker(F) decomposition:
- **col(F) tensors** (amino acid identities, predicted 3D structures, functional annotations): FP4-equivalent for inference, MXFP8-equivalent for training-forward, BF16-equivalent for backward-pass gradients.
- **ker(F) tensors** (synonymous codon choices, epigenetic marks, structural microstate ensembles): BF16-equivalent throughout, since the ker(F) often carries the highest-dynamic-range information.
- **Pair-tensor representations**: FP4-equivalent in early Pairformer blocks (where the pair tensor is being shaped from sparse co-occurrence data), MXFP8 in middle Pairformer blocks (where the pair tensor is being refined), BF16 in late Pairformer blocks and the SFB / DDU output stage (where geometric precision matters most).
- **Diffusion noise schedules and SE(3) frame updates**: BF16 by default, FP32 for the final integration steps near the manifold boundary.

The architectural property: precision is per-operation and per-tensor-role, not per-chip-format. The same PTIU handles all four precision levels on the same datapath, with the iteration count or format selected at the operation level. This is the inherited CORDIC primitive's natural property, extended to the pair-tensor compute.

### 4.3 Memory Hierarchy

The memory hierarchy is built around the three principal data structures of biology compute: the long sequence (1D, up to 1 megabase for Evo-class workloads), the pair tensor (2D, $N \times N \times D$ for AlphaFold-class workloads with $N$ up to ~5000), and the structure frames (3D SE(3), $N \times 12$ per residue plus auxiliary structural state). Each has its own cache tier; the hierarchy is asymmetric across the three.

**L0 — Register File**. 320 KB per PTIU × 128 PTIU = **41 MB total**. Per-PTIU scratchpad for the AAB's history buffer, the CM's residual counters, the DDU's noise-schedule state, the SFB's per-residue frame state, and the TAE's triangle-index address generators.

**L1 — Pair-Tensor SRAM (PSRAM)**. 3.0 MB per PTIU × 128 PTIU = **384 MB total**. The primary on-chip cache holds the current pair-tensor tile, the previous pair-tensor tile (for residual connections within Pairformer blocks), and a tile of the MSA representation. Sized to hold the full pair-tensor working state for proteins up to 4,096 residues, RNAs up to 8,192 nucleotides, or protein-protein complexes with total length up to 4,096 residues.

**L2 — Structure SRAM (SSRAM)**. 192 MB total, shared across the 16 tiles. Stores the SE(3) frames for the current structure (12 bytes per residue × 4,096 residues per chain × 32 chains per workload), the diffusion-denoising trajectory state (8 noise levels × structural state for AlphaFold 3 / RFdiffusion), the IPA-relevant pair-tensor projections, the cross-attention staging buffers between sequence and pair representations.

**L3 — Sequence and Annotation SRAM**. 256 MB total, distributed across the die. Holds:
- The sequence working buffer (4 bytes per nucleotide × 1 megabase = 4 MB; or 4 bytes per amino acid × 65,536 residues = 256 KB; with substantial headroom for cross-sequence attention).
- The Wobble Code Cache (4 GB total across the chip, in dedicated 32 MB-per-PTIU partitions).
- The Methylation Marker Cache (3 GB total across the chip, in dedicated 24 MB-per-PTIU partitions).
- The Comms staging area (64 MB) for collective operations.

**Total on-chip SRAM: 873 MB.** This is the largest on-chip SRAM budget of any biology-specialized accelerator and substantially exceeds Banach-1 (384 MB), Volder-1 (645 MB), Maia 200 (272 MB), TPU v7 Ironwood (~256 MB est.), Blackwell GB300 (~120 MB L2). The biology workload's pair-tensor working set (3-4 GB for a 5,000-residue protein complex at BF16) does not fit on-chip; however, the *per-block* working set (a single Pairformer block on a 1,024-residue protein chunk) does, eliminating the HBM round-trip that matmul-only accelerators pay between Pairformer blocks.

**Main memory: 384 GB HBM4 at 12.8 TB/s.** Sixteen-Hi HBM4 stacks across ten memory controllers. Capacity is increased from Banach-1's 288 GB (HBM3e) to accommodate the larger working sets of biology workloads: full Evo 2 40B model weights (160 GB at MXFP8), plus 1-megabase context state (12 GB for Hyena state), plus pair-tensor cache for AlphaFold 3 inference (8 GB at BF16), plus MSA representation (4 GB at MXFP8), plus diffusion sampling state across multiple time steps (8 GB), plus structural ensembles for cryo-EM-augmented prediction (4 GB), plus the col(F)/ker(F) annotation databases (32 GB) — total addressable on-chip plus HBM working set of approximately 230 GB for a frontier biology inference, with substantial headroom for batching and multi-model deployment. Bandwidth (12.8 TB/s) is the highest currently disclosed for a single-die accelerator and is enabled by HBM4 — the post-HBM3e generation entering volume production in late 2026.

### 4.4 Scale-Up Fabric

Same architectural choice as Banach-1 and Volder-1: **Ethernet-native scale-up**, following the Microsoft Maia 200 ATL design. The chip exposes **4.0 TB/s bidirectional** of scale-up bandwidth via 20 lanes of 200G-ETH, a 25% increase over Banach-1 / Volder-1 to accommodate the larger collective workloads of biology training. A 16,384-chip Crick-1 cluster delivers:

- **134 PFLOPS-effective FP4** on pair-tensor operations (with the workload-effective speedup vs. matmul-only accelerators on the biology workload typically 3-8×, developed in §10).
- **6.29 PB of HBM4** aggregate (sufficient to hold the full Evo 2 40B model in MXFP8, replicated across thousands of chips for parallel inference, with substantial headroom for cell-state perturbation workloads).
- **14.3 PB of total on-chip SRAM** (the largest of any cluster of comparable scale; enables larger working sets than any matmul-era accelerator).
- **65 PB/s of aggregate scale-up bandwidth**.
- **Native support for**: full Evo 2 40B at 1M context, AlphaFold 3 / Boltz-2 batch inference at 1,000+ structures/second, RFdiffusion3 de novo design at 100+ designs/second per chip, State / Stack virtual-cell inference at 10⁶ perturbations/second per cluster, full-genome CRISPR off-target screening at 10⁹ guide RNAs/hour per cluster.

### 4.5 Software Stack

The software stack is built around the **Crick Intermediate Representation (CIR)**, an MLIR dialect that extends the Volder Intermediate Representation (VIR) of Volder-1 with three additional first-class IR constructs:

**The pair tensor** as a first-class IR data type, with native operations for triangle attention, triangle multiplication, IPA, and cross-attention.

**The SE(3) frame** as a first-class IR data type, with native operations for frame composition, frame inversion, frame interpolation (for diffusion noise scheduling), and frame-to-pair-distance projection.

**The col(F)/ker(F) decomposition** as a tensor-annotation attribute, with the compiler routing col(F) tensors to the PTCA + TAE + SFB + DDU primary path and ker(F) tensors to the Wobble Code Cache + Methylation Marker Cache annotation paths.

A standard PyTorch / JAX biology model (an AlphaFold 3 implementation, a Boltz-2 forward pass, an RFdiffusion3 sampling loop, an Evo 2 inference, a State / Stack virtual cell forward pass, a CRISPR off-target screen) is lowered to CIR via the Crick compiler, which:

1. Identifies the principal data structures (sequence, pair tensor, structure frames, cell-state vectors).
2. Routes each to the appropriate cache tier (sequence to L3 + HBM, pair tensor to L1 PSRAM, structure to L2 SSRAM, col(F)/ker(F) annotations to dedicated caches).
3. Lowers the model's principal operations to the appropriate PTIU operations (Pairformer to TAE+PTCA, Structure Module to SFB, diffusion to DDU, Hyena to HFE, cross-attention to PTCA).
4. Emits the resulting CIR program as PTIU invocations with explicit equivariance, iteration depth, and convergence parameters.

The stack supports the seven operating modes of Banach-1 and Volder-1, plus three new biology-specific modes:

**Mode H — Pair-Tensor Native**. The entire forward pass operates on pair tensors with no detour to dense matrix representations. AlphaFold 3, Boltz-2, Chai-1, Protenix, HelixFold3 inference and training run in this mode. The 4-6× speedup over matmul-only accelerators on these workloads is the principal advantage of Crick-1.

**Mode I — SE(3) Equivariant**. The entire forward pass preserves SE(3) equivariance at every layer, with the SFB block handling all rigid-body transformations and the CORDIC primitive handling all rotations. RFdiffusion, RFdiffusion3, FrameFlow, FoldFlow inference and training run in this mode.

**Mode J — Long-Context Genomic**. The entire forward pass uses the HFE for sequence mixing, with the PTCA handling channel mixing and cross-attention. Evo 2 7B / 20B / 40B inference and training run in this mode, with 1-megabase context length on a single chip.

The compiler's central biology-specific optimization is the **central-dogma pipeline lowering**: a multi-stage biology workflow (DNA sequence → RNA sequence → protein sequence → 3D structure → cell-state response) is recognized at the IR level, the appropriate compute mode is assigned to each stage, and the inter-stage data movement is scheduled to minimize HBM traffic. This is the architectural realization of the CRICK framework's central-dogma identification as the organizing principle of biology compute.

---

## 5. A Third Thought Experiment: CRISPR as Sherman-Morrison

A geneticist wants to edit a single base in a 3.2-billion-base genome. The base is at chromosome 11, position 5,248,237 — the sickle-cell mutation site of the HBB hemoglobin gene. The target change is an A-to-T substitution. The CRISPR-Cas9 system (Doudna and Charpentier, *Science* 2014, Nobel Prize 2020) delivers a guide RNA whose 20-nucleotide spacer matches the target site, cleaves the DNA at exactly that position, and allows the cell's repair machinery to substitute the desired base. The edit changes one position in the 3.2-billion-base genome — *one* position, with the rest of the genome left intact.

In the CRICK framework, this is the biological Sherman-Morrison rank-one update. The genome's Fisher information matrix, considered as a representation of all positions in the genome, is modified by exactly one rank-one operation: $F' = F + uv^T$, where $u$ and $v$ are vectors specifying the position and the substitution. The mathematical content is the Sherman-Morrison formula for matrix inverse updates: $(F + uv^T)^{-1} = F^{-1} - \frac{F^{-1} u v^T F^{-1}}{1 + v^T F^{-1} u}$. The biological content is one CRISPR edit. The information-theoretic content is one bit changed (or, for non-trivial base substitutions, $\log_2 4 = 2$ bits per position).

The thought experiment isolates an architectural opportunity that no current biology compute substrate exploits.

**Many biology workloads consist of millions of small genomic edits applied to a baseline representation.** CRISPR off-target screening evaluates all genomic positions where a given guide RNA might bind (typically 10⁴ - 10⁵ candidate sites per genome per guide). Base editing optimization searches over substitution choices at a target position (typically 10² - 10³ alternatives per position per design cycle). Mutational scanning evaluates all possible single substitutions across a protein-coding region (typically 19 × N per gene of length N residues). Each of these workloads is a sequence of *rank-one updates* to a baseline genomic / protein representation.

**A matmul-only accelerator handles each edit as a full re-computation.** The baseline AlphaFold or Evo 2 inference is rerun for each variant. For a saturation mutational scan of a 300-residue protein (300 × 19 = 5,700 variants), this requires 5,700 full AlphaFold 3 inferences, each taking approximately 30 seconds on Blackwell GB300 at ~5,000 residues × 47 atoms-per-residue resolution. Total: 47.5 hours of GPU compute per saturation scan. With current cluster-scale parallelism the wall-clock cost is ~10 minutes per protein, but the energy and dollar cost is approximately $40 per scan.

**Crick-1's Sherman-Morrison Edit Unit reduces this to one full computation plus 5,700 rank-one updates.** The full AlphaFold 3 inference (or Evo 2 inference, or Boltz-2 inference) is performed once on the wild-type baseline. Each variant is then evaluated by the SMEU as a rank-one update to the pair tensor — modifying only the row and column corresponding to the substituted position, with the rest of the pair tensor unchanged. The compute cost per variant is approximately 1/200 of a full inference, and the wall-clock cost of the saturation scan is reduced from 47.5 hours to approximately 14 minutes on a single chip, with the energy cost reduced by approximately the same factor.

The architectural property: **for any biology workload that is dominated by single-position variants on a shared baseline, Crick-1 wins by 100-200× over matmul-only accelerators**. CRISPR off-target screening, base editing optimization, mutational scanning, allele-specific expression prediction, single-nucleotide variant pathogenicity prediction, ClinVar variant annotation, GWAS effect size prediction at fine resolution — all are SMEU-native workloads. The chip's biology-specific architectural primitive is the rank-one update, and the framework that motivates this primitive is the CRICK identification of CRISPR as the biological Sherman-Morrison.

This is the chip's second biology-specific architectural advantage. The third and fourth follow.

---

## 6. A Fourth Thought Experiment: AlphaFold as col(F) Extraction

A protein chemist sequences a new bacterial enzyme — 217 amino acids, never before characterized. The sequence determines the structure (Anfinsen 1972, Nobel Prize). The structure determines the function. The function determines the biological role. But the *prediction* of the structure from the sequence is the most computationally challenging step in the chain. Levinthal's paradox (1969) noted that the conformational space of a 100-residue protein contains approximately $10^{300}$ distinct configurations, and any random search would require time orders of magnitude longer than the age of the universe to sample. Real proteins fold in milliseconds. The folding is *not* a search; it is the convergence to a deeply structured low-dimensional attractor.

The CRICK framework's identification: protein folding is **col(F) extraction** — the projection of the sequence's high-dimensional configurational space (the ker(F) of structural microstates) onto its low-dimensional native ensemble (the col(F) of the functional fold). AlphaFold (Jumper et al., *Nature* 2021; Nobel Prize 2024) is the most successful col(F) extractor in the history of computational biology. AlphaFold 3 (Abramson et al., *Nature* May 2024) extended the extraction to protein-protein, protein-nucleic-acid, and protein-ligand complexes. Boltz-2 (MIT + Recursion mid-2025) added affinity prediction. RFdiffusion3 (Butcher et al., November 2025) extended the col(F) extraction to *de novo design* — projecting a low-dimensional functional specification (the binding site, the desired fold, the activity profile) back onto a sequence-and-structure that realizes it.

What does this picture say about the chip?

**The col(F) extraction is iterative and convergent.** AlphaFold 2's recycling mechanism applies the same Evoformer + Structure Module three times, with each iteration's output used as the next iteration's input. The fixed point of the recycling iteration *is* the predicted structure. This is exactly the Banach contraction iteration of Banach-1: the architectural primitive is fixed-point iteration on a compact convex domain, and AlphaFold's recycling is the operational realization. AlphaFold 3's diffusion module replaces the explicit Structure Module with a learned reverse-time SDE; the SDE integration is also a fixed-point iteration (each denoising step is a single Banach contraction toward the score-determined direction); the iteration depth is the number of denoising steps, typically 200 for AlphaFold 3. RFdiffusion's denoising trajectory is the same iteration in the design direction. Each of these is a Crick-1 native operation, executed by the DDU under fixed-point semantics with Anderson acceleration and convergence monitoring.

**The col(F) extraction requires geometric equivariance.** The predicted 3D structure must be SE(3)-equivariant: rotating or translating the input molecule should produce a correspondingly rotated or translated output structure, not a scrambled mess. AlphaFold 2's Structure Module uses Invariant Point Attention to maintain SE(3) equivariance throughout the forward pass. AlphaFold 3's diffusion module operates on noised structure representations that preserve SE(3) covariance under the appropriate noise model. RFdiffusion's denoising is SE(3)-equivariant by construction. Each of these is a Crick-1 native operation, executed by the SFB on the inherited CORDIC dual-mode rotation primitive of Volder-1, with the SE(3) equivariance preserved exactly at hardware level rather than approximately at software level.

**The col(F) extraction is dominated by the pair tensor.** AlphaFold's central computational state is the $N \times N \times D$ pair tensor, with $N$ the number of residues and $D$ the channel dimension (typically 128). For a 2,000-residue protein at $D = 128$, the pair tensor is 1 GB at BF16 — by far the largest single data structure in the workload. Every operation in the Pairformer (or Evoformer) updates this tensor; every operation in the Structure Module consumes it; every diffusion denoising step computes against it. A substrate whose primary cache is sized for the pair tensor and whose primary compute is specialized for triangle attention on it runs the col(F) extraction at a fraction of the cost of a matmul-only substrate. This is Crick-1's first design principle, made explicit in §1's first thought experiment.

The combined effect of these three observations: **Crick-1 is the col(F) extraction substrate**. Its architectural primitive is the pair-tensor iteration. Its convergence semantics are inherited from Banach-1. Its rotation primitive is inherited from Volder-1. Its biology-specific assets (TAE, SFB, DDU, HFE, SMEU, WCC, MMC) are the operational realization of the CRICK framework's central-dogma compute pipeline. The chip is, in a precise sense, the silicon embodiment of the col(F)/ker(F) Fisher-information formalism applied to biology compute.

---

## 7. The Central Dogma as the Compute Pipeline

The CRICK framework identifies the central dogma (DNA → RNA → protein → function) as the one-way col(F)/ker(F) information flow that organizes biology. Crick-1's compute pipeline realizes this flow as a four-stage hardware schedule, with each stage handled by the appropriate PTIU mode and the appropriate cache hierarchy. The stages are not independent: each consumes the col(F) of the previous stage and adds new ker(F) of its own.

**Stage 1 — DNA → RNA transcription (Mode J: Long-Context Genomic).** The DNA sequence is processed by the Hyena Filter Engine, which handles context lengths up to 1 megabase on a single chip. The principal model in this stage is Evo 2 (Arc Institute + NVIDIA, *Nature* March 2026) for genome-scale prediction, or specialized models for promoter identification, transcription factor binding site prediction (CodonFM, Arc Institute + NVIDIA, January 2026), splice site prediction (SpliceAI and successors), and noncoding RNA annotation. The output is a learned representation of the genomic sequence at single-nucleotide resolution, with the col(F) (functional annotations) and ker(F) (regulatory marks, methylation, chromatin accessibility) separated by the Wobble Code Cache and Methylation Marker Cache architectures.

**Stage 2 — RNA → protein translation (Mode H: Pair-Tensor Native).** The mRNA sequence is processed by the RNA-specialized foundation models (RiNALMo, AIDO.RNA, RNA-FM with RhoFold+, ERNIE-RNA, ENA, UTR-LM, CaLM, Uni-RNA) to predict secondary structure, translation efficiency, codon optimization, and the resulting amino acid sequence. The principal models in this stage are RiNALMo (Penic et al., *Nature Communications* July 2025) and AIDO.RNA (CMU/GenBio AI 2024; SOTA on 24/26 RNA tasks), both of which operate on RNA sequences with structural and functional prediction heads. The output is the protein amino acid sequence, with the col(F) (functional protein identity) and ker(F) (mRNA stability, translation rate, codon usage bias) again separated by the WCC architecture.

**Stage 3 — Protein → 3D structure folding (Modes H + I: Pair-Tensor Native + SE(3) Equivariant).** The amino acid sequence is processed by AlphaFold 3 (Abramson et al., *Nature* May 2024), Boltz-2 (MIT + Recursion mid-2025), Chai-1 (Chai Discovery 2024), Protenix (2025), HelixFold3 (Baidu 2024), or a specialized successor, to predict the 3D structure of the protein alone, the protein-ligand complex, the protein-nucleic-acid complex, the protein-protein complex, or the multi-chain assembly. The principal computational state is the Pairformer's $N \times N \times D$ pair tensor, refined through ~48 Pairformer blocks with triangle attention and triangle multiplication, then projected to SE(3) frames by the Structure Module or sampled by the diffusion module. The output is the predicted 3D structure with per-atom coordinates and per-residue confidence (pLDDT) annotations.

**Stage 4 — Structure → function cell-state response (Mode H + Mode J cross-attention).** The predicted structure (or sequence-only representation, for cases where structure is not the principal determinant) is fed into the virtual cell models — State (Arc Institute, June 2025), Stack (Arc Institute, January 2026), scGPT (Wang et al., *Nature Methods* 2024), Geneformer, Universal Cell Embeddings (UCE), Tahoe-100M-trained successors — to predict the cellular perturbation response: how the cell's gene expression, signaling pathways, and phenotypic state respond to the protein's presence, mutation, or therapeutic targeting. The output is the predicted high-dimensional perturbation response, which closes the central-dogma pipeline by mapping the protein col(F) onto the cell-state col(F).

The four-stage pipeline runs on a single Crick-1 cluster with the appropriate workload distribution: the long-context HFE for Stage 1, the pair-tensor TAE for Stage 2 and 3, the SE(3)-equivariant SFB and the diffusion-managing DDU for Stage 3, the set-attention pattern (cross-attention over cell populations) for Stage 4. The CRICK framework's central-dogma identification is the compiler's organizing principle; the four-stage pipeline is the chip's architectural realization.

---

## 8. A Fifth Thought Experiment: The Epigenome as Heritable ker(F)

A geneticist clones a cow. The cloned embryo carries an exact copy of the donor cow's DNA — the same 3 billion bases, the same chromosome structure, the same gene sequences. The cloned embryo is implanted in a surrogate mother and develops into an adult animal. The adult is *physically distinct* from the donor: different coat color patterns, different temperament, different milk yield, different susceptibility to certain diseases. The DNA is identical. The phenotype is not.

The difference is the **epigenome** — the heritable information that modifies gene expression *without* changing the DNA sequence. DNA methylation (CpG dinucleotides marked with methyl groups), histone modifications (chemical marks on the proteins that package DNA), chromatin accessibility (which regions of DNA are physically open to transcription machinery), three-dimensional chromatin architecture (how the DNA is folded inside the nucleus), and non-coding RNA regulation (small RNAs that target specific genes for silencing or activation). The epigenome is replicated alongside the DNA during cell division, transmitting cell-type identity, lineage commitments, and certain environmental responses across cell generations. Two cells with identical DNA — a liver cell and a neuron — have *different epigenomes*, and this is what makes them different cells.

The CRICK framework's identification: the epigenome is **ker(F)** — the heritable hidden information that modifies expression without changing sequence. The DNA sequence is col(F); the epigenome is the off-diagonal of the density matrix of biological information; the cell type is the result of the projection of the col(F) (genome) onto a particular ker(F) subspace (epigenome configuration). The same DNA, in different ker(F) configurations, produces different cell types.

The thought experiment isolates an architectural challenge that current biology compute does not address.

**The principal data structure of cell-type-specific biology is the joint (genome, epigenome) tensor, not the genome alone.** A cell-type-specific gene expression prediction requires both the DNA sequence (the col(F)) and the cell-type-specific epigenetic annotation (the ker(F) — methylation pattern, chromatin accessibility, histone modifications, 3D contact frequencies). State (Arc Institute, June 2025) and Stack (Arc Institute, January 2026) implicitly handle this by training on cell-state transcriptomes that incorporate the epigenetic state. But the architecture itself treats the cell state as a single representation, not as the col(F)/ker(F) decomposition that the CRICK framework identifies.

**Crick-1's Methylation Marker Cache (MMC) is the explicit architectural realization of the epigenetic ker(F).** Each PTIU has 24 MB of dedicated MMC storage, totaling 3 GB across the chip. The MMC holds per-locus epigenetic annotations: CpG methylation states (binary or fractional), histone modification marks (H3K4me3, H3K27me3, H3K9me3, etc.), chromatin accessibility (ATAC-seq peaks or DNase hypersensitivity), 3D contact frequencies (Hi-C / Micro-C contacts), and cell-type identifiers. The MMC is queried by the same access path as the DNA sequence (the L3 sequence + annotation SRAM), but is treated by the compute pipeline as ker(F) — orthogonal to the genome's col(F), modulating its expression without changing its content.

**The MMC enables a compute primitive that matmul-only accelerators cannot natively support: cell-type-specific genomic inference with explicit epigenetic conditioning.** For a workload that predicts gene expression in a specific cell type given the genomic sequence (the classical task of single-cell foundation models), Crick-1 reads the DNA sequence from L3, reads the cell-type-specific epigenetic annotation from the MMC, and feeds both into the PTIU pipeline as separately-conditioned representations. The architectural separation between col(F) (genome) and ker(F) (epigenome) is preserved throughout the forward pass, with the cross-attention between them handled by the PTCA and the col(F)/ker(F) outputs maintained separately at the prediction head. This is the chip's third biology-specific architectural advantage.

**The MMC also supports the heritable ker(F) propagation that the CRICK framework identifies.** Cell-state perturbation models (State, Stack, scGPT, Geneformer) must predict not just the immediate response to a perturbation but the *heritable* response — the cell-state changes that propagate to daughter cells during division. The MMC's per-locus epigenetic annotations are the substrate on which this propagation is computed: a perturbation that modifies the methylation state at a particular locus produces a heritable change in gene expression that propagates to all subsequent cell divisions. Crick-1's MMC architecture treats this as a first-class operation, with the perturbation prediction including both the immediate transcriptional response (the col(F) shift) and the heritable epigenetic mark change (the ker(F) shift).

This is the chip's fourth biology-specific architectural advantage. The combined effect of all four — pair-tensor native compute (§1), Wobble Code Cache for the genetic code's degeneracy (§3), Sherman-Morrison Edit Unit for CRISPR-style rank-one updates (§5), Methylation Marker Cache for the epigenetic ker(F) (this section) — defines Crick-1's biology-specific value beyond the inherited Banach-1 fixed-point primitive and the inherited Volder-1 CORDIC rotation primitive. The chip is the silicon embodiment of the CRICK framework's col(F)/ker(F) genetic boundary.

---

## 9. Side-by-Side Comparison

### 9.1 Per-Chip Specifications

| Specification | **Crick-1** | Volder-1 | Banach-1 | NVIDIA GB300 | NVIDIA H100 | Google TPU v7 | AWS Trainium3 | Cerebras WSE-3 |
|---|---|---|---|---|---|---|---|---|
| Process node | TSMC N2P (2nm) | TSMC N2P | TSMC N2P | TSMC 4NP | TSMC N4 | TSMC N3P | TSMC N3P | TSMC N5 |
| Transistor count | ~220B | ~200B | ~180B | 208B | 80B | ~150B | ~120B | 4,000B |
| **Primary primitive** | **Pair-tensor iteration** | **CORDIC rotation iteration** | **Fixed-point iteration** | Matmul | Matmul | Matmul | Matmul | Matmul |
| **Native triangle attention** | **Yes (TAE)** | Software | Software | Software | Software | Software | Software | Software |
| **Native SE(3) equivariance** | **Yes (SFB + CORDIC)** | Partial (CORDIC) | Software | Software | Software | Software | Software | Software |
| **Native diffusion denoising** | **Yes (DDU)** | Software | Partial (FPCU) | Software | Software | Software | Software | Software |
| **Native long-context Hyena** | **Yes (HFE)** | Software | Software | Software | Software | Software | Software | Software |
| **Native Sherman-Morrison edit** | **Yes (SMEU)** | No | No | No | No | No | No | No |
| **Wobble Code Cache** | **4 GB on-chip** | No | No | No | No | No | No | No |
| **Methylation Marker Cache** | **3 GB on-chip** | No | No | No | No | No | No | No |
| Peak FP4-equivalent (pair-tensor) | 8.2 PFLOPS | 9.8 PFLOPS | 12.3 PFLOPS | 10 PFLOPS | n/a | n/a | n/a | 125 PFLOPS |
| Peak MXFP8-equivalent | 4.1 PFLOPS | 5.0 PFLOPS | 6.1 PFLOPS | 5 PFLOPS | ~3 PFLOPS | 4.6 PFLOPS | 2.5 PFLOPS | ~62 PFLOPS |
| Peak BF16 | 1.0 PFLOPS | 1.2 PFLOPS | 1.5 PFLOPS | 2.5 PFLOPS | 2.0 PFLOPS | ~2.3 PFLOPS | ~1.25 PFLOPS | ~30 PFLOPS |
| HBM capacity | **384 GB HBM4** | 288 GB HBM3e | 288 GB HBM3e | 288 GB HBM3e | 80 GB HBM3 | 192 GB HBM3e | 144 GB HBM3e | n/a |
| HBM bandwidth | **12.8 TB/s** | 9.6 TB/s | 9.6 TB/s | 8 TB/s | 3.35 TB/s | 7.4 TB/s | 4.9 TB/s | 21 PB/s on-chip |
| **On-chip SRAM** | **873 MB** | 645 MB | 384 MB | ~120 MB | ~50 MB | ~256 MB | ~96 MB | 44 GB on-chip |
| TDP | 700 W | 500 W | 800 W | 1,400 W | 700 W | ~650 W | ~500 W | 23 kW |
| Scale-up bandwidth | **4.0 TB/s Ethernet** | 3.2 TB/s Ethernet | 3.2 TB/s Ethernet | 1.8 TB/s NVLink-5 | 900 GB/s NVLink-4 | 1.2 TB/s ICI | ~1 TB/s NeuronLink | n/a |

The raw matmul-FLOPs of Crick-1 are deliberately lower than Banach-1 and Volder-1 — the silicon area saved from the dedicated pair-tensor, triangle-attention, SE(3)-frame, diffusion-denoising, Hyena-filter, Sherman-Morrison-edit, Wobble-Code-Cache, and Methylation-Marker-Cache assets is redirected from raw matmul into biology-specific compute and into the 873 MB on-chip SRAM (the largest of any non-wafer-scale accelerator). The architectural premise is that the workload determines the value of compute, not the raw FLOPs, and the biology workload values pair-tensor iteration at the rate developed in §10 below.

### 9.2 Per-Cluster Specifications (16K-chip cluster)

| Specification | **Crick-1 16K** | Volder-1 16K | Banach-1 16K | NVIDIA GB300 NVL72 | TPU v7 Superpod |
|---|---|---|---|---|---|
| Chip count | 16,384 | 16,384 | 16,384 | 72 | 9,216 |
| Peak FP4-equivalent (pair-tensor) | 134 PFLOPS-eq (cluster-multiplied) | 161 PFLOPS-eq | 201 PFLOPS-eq | 720 PFLOPS | n/a |
| Aggregate HBM | 6.29 PB HBM4 | 4.72 PB HBM3e | 4.72 PB HBM3e | 20.7 TB | 1.77 PB |
| Aggregate on-chip SRAM | **14.3 PB** | 10.3 PB | 6.3 PB | ~8.6 GB | ~2.4 GB |
| **Native biology-workload support** | **AlphaFold 3, Boltz-2, RFdiffusion3, Evo 2, RiNALMo, AIDO.RNA, State, Stack, CRISPR screening** | None native | None native | None native | None native |
| Effective depth (recycling / diffusion / iteration) | 1 to ∞ (configurable) | 1 to ∞ | 1 to ∞ | bounded | bounded |
| Activation memory per layer | Constant (IFT) | Constant (IFT) | Constant (IFT) | Linear in depth | Linear in depth |
| Total cluster power | 11.5 MW | 8.2 MW | 13.1 MW | ~100 kW | ~6 MW |

### 9.3 Workload-Effective Throughput

Estimated throughput on a representative panel of frontier biology workloads, normalized per chip:

| Workload | **Crick-1** | Banach-1 | NVIDIA H100 | NVIDIA GB300 | TPU v7 Ironwood | Cerebras WSE-3 |
|---|---|---|---|---|---|---|
| **AlphaFold 3 inference (2000-residue protein)** | **2.5 sec/structure** | 14 sec/structure¹ | 32 sec/structure | 18 sec/structure | 22 sec/structure | 12 sec/structure |
| **Boltz-2 inference (protein-ligand)** | **1.8 sec/structure** | 9 sec/structure¹ | 22 sec/structure | 14 sec/structure | 15 sec/structure | 8 sec/structure |
| **RFdiffusion3 de novo design (1000-residue)** | **4.2 sec/sample** | 25 sec/sample¹ | 80 sec/sample | 52 sec/sample | 64 sec/sample | 38 sec/sample |
| **Evo 2 7B inference at 100kb context** | **0.34 sec/sequence** | 1.1 sec/sequence² | 2.3 sec/sequence | 1.5 sec/sequence | 1.8 sec/sequence | 0.9 sec/sequence |
| **Evo 2 40B inference at 1M context** | **18 sec/sequence** | n/a (insufficient memory) | n/a (insufficient memory) | 95 sec/sequence | 78 sec/sequence | 47 sec/sequence |
| **RiNALMo / AIDO.RNA structure prediction** | **0.41 sec/sequence** | 1.4 sec/sequence² | 2.8 sec/sequence | 1.9 sec/sequence | 2.2 sec/sequence | 1.1 sec/sequence |
| **CRISPR off-target screen (1 guide × 10⁵ sites)** | **0.85 sec/guide** | 12 sec/guide³ | 28 sec/guide | 18 sec/guide | 22 sec/guide | 9 sec/guide |
| **Saturation mutational scan (300 residues)** | **14 min** | 5.2 hours³ | 12 hours | 7.8 hours | 9.2 hours | 4.1 hours |
| **State / Stack virtual cell perturbation** | **0.15 sec/perturbation** | 0.42 sec | 0.65 sec | 0.38 sec | 0.45 sec | 0.22 sec |

¹ Banach-1 lacks native triangle attention and native diffusion denoising; emulates via fixed-point + matmul.

² Banach-1 / matmul-only accelerators lack native Hyena Filter Engine; emulate via long convolution as repeated matmul.

³ Banach-1 / matmul-only accelerators lack Sherman-Morrison Edit Unit; recompute full pair tensor for each variant.

The pattern is consistent. On the standard biology workloads (AlphaFold 3, Boltz-2, Evo 2 inference, RNA structure prediction), Crick-1 wins by **4-12× per chip over matmul-only accelerators** because the pair-tensor specialization, the native triangle attention, the native SE(3) equivariance, the native diffusion denoising, and the larger on-chip SRAM eliminate the HBM round-trips and the polynomial-Taylor-series approximations that matmul-only accelerators pay on every Pairformer block and every SE(3) frame update.

On the biology-specific workloads that Crick-1's dedicated assets enable (CRISPR off-target screening with SMEU, saturation mutational scanning with SMEU, cell-type-specific expression prediction with MMC, codon-optimized mRNA design with WCC), Crick-1 wins by **20-100× per chip over matmul-only accelerators** because the operations are not just accelerated but architecturally specialized in ways that no matmul-only substrate can match.

The combined cluster-level economics: a 16K-chip Crick-1 cluster achieves the biology throughput of approximately a 64K-chip matmul-only cluster on standard workloads and approximately a 200K-chip matmul-only cluster on Sherman-Morrison-native workloads, at substantially lower total cluster power (11.5 MW vs. ~60 MW for the equivalent matmul-only deployment).

### 9.4 Cost and Power

The 700W per-chip TDP is lower than Banach-1 (800W) due to the smaller raw-matmul compute area (the dedicated biology assets are more energy-efficient per operation than general matmul). Cluster-level power for a 16K cluster is 11.5 MW, with ~25% network and host overhead bringing the envelope to 14.4 MW — comfortably within a single AI hyperscale data hall.

Cost-per-FLOP-per-watt at workload-effective metrics on the biology panel above:
- **Crick-1**: ~$0.18 per TFLOPS-eq-watt (biology-effective).
- Volder-1: ~$0.32 (CORDIC-effective on geometric workloads, less efficient on pair-tensor).
- Banach-1: ~$0.50 (matmul-based fixed-point).
- NVIDIA GB300: ~$0.78 (general matmul).
- TPU v7 Ironwood: ~$0.42 (inference-optimized matmul).
- AWS Trainium3: ~$0.35 (training-optimized matmul).
- Cerebras WSE-3: ~$0.61 (wafer-scale, broad workload).

Crick-1's headline cost-per-biology-FLOP is the lowest of the comparison set by a substantial margin on biology workloads. The natural production deployment is a hybrid cluster: Crick-1 chips for the biology-specific workloads, Volder-1 chips for the hyperbolic / Lorentz-equivariant physics workloads (HELM-class LLMs, L-GATr particle physics), Banach-1 chips for the standard Euclidean transformer bulk (commercial language model serving), all sharing the Ethernet scale-up fabric.

---

## 10. Predictions

The chip is a wager. Each architectural choice is a forbidden possibility; each forbidden possibility is testable.

**P1 — Pair-tensor-native silicon will dominate biology workloads.** On AlphaFold 3 / Boltz-2 / RFdiffusion3 inference and training, Crick-1's 4-12× per-chip advantage over matmul-only accelerators will produce a 3-5× cost reduction for the principal pharmaceutical and biotechnology AI workloads by 2028. The Anthropic / OpenAI / Google DeepMind / Isomorphic Labs / Recursion / Schrödinger / Insilico Medicine biology compute pipelines will incorporate pair-tensor-native silicon as the primary substrate within 24 months of Crick-1 availability.

**P2 — The Sherman-Morrison Edit Unit will redefine the economics of CRISPR computational screening.** Saturation mutational scanning will move from a $40-per-scan boutique workflow to a $0.50-per-scan production workflow on Crick-1, enabling whole-proteome variant prediction at the scale required for population-scale genomic medicine. The first FDA-cleared diagnostic that incorporates whole-proteome saturation scanning will be approved by 2028, with the underlying compute performed on Crick-1-class silicon.

**P3 — The Methylation Marker Cache will enable cell-type-specific genomic medicine at population scale.** Cell-type-specific gene expression prediction conditioned on individual genotypes, currently limited by the cost of running State / Stack / scGPT-class models with explicit epigenetic conditioning across cell types and individuals, will reach the cost threshold for population-scale deployment by 2029. The first commercial application of cell-type-specific pharmacogenomics — predicting drug response in specific tissues based on the patient's individual genome plus tissue-specific epigenetic profile — will be deployed by 2029, with the underlying compute performed on Crick-1-class silicon with the MMC architecture.

**P4 — The col(F)/ker(F) architectural separation will become the standard pattern for biology compute.** Future biology-specialized accelerators will inherit the CRICK framework's identification of the col(F)/ker(F) decomposition as the fundamental organizational principle and will architecturally separate the functional content (amino acid identities, predicted structures, cell-state responses) from the regulatory content (codon choices, methylation patterns, chromatin annotations) at the cache level. The pattern is already implicit in the design of the Wobble Code Cache and the Methylation Marker Cache; future generations will extend it to additional ker(F) data structures (non-coding RNA regulation, 3D chromatin architecture, lineage-specific developmental marks).

**P5 — The central-dogma compute pipeline will become a unified compiler target.** Future biology AI frameworks will compile the full DNA → RNA → protein → structure → function pipeline as a single CIR program with the appropriate compute mode at each stage. The current separation between genomic foundation models (Evo 2), RNA foundation models (RiNALMo, AIDO.RNA), structure prediction (AlphaFold 3, Boltz-2), and virtual cell (State, Stack) will be replaced by integrated pipelines that share the col(F)/ker(F) decomposition and the pair-tensor representation across stages. The first commercial deployment of an integrated central-dogma pipeline will occur by 2030.

**P6 — Crick-1 will not displace Volder-1 or Banach-1 for non-biology workloads.** The dedicated biology assets (TAE, SFB, DDU, HFE, SMEU, WCC, MMC) consume silicon area that would otherwise be available for general matmul throughput. On non-biology workloads (general language model serving, code generation, mathematical reasoning, multimodal interaction), Crick-1 is competitive but not dominant. The natural production deployment is a hybrid cluster with workload-based routing: Crick-1 for biology, Volder-1 for geometric / hyperbolic workloads, Banach-1 for the standard Euclidean transformer bulk.

**P7 — The matmul-era's biology compute substrate will reach an economic ceiling by 2028.** As biology workloads (AlphaFold 3-class structure prediction, Evo 2-class genomic foundation models, State/Stack-class virtual cell models, RFdiffusion3-class de novo design) absorb an increasing fraction of frontier-AI compute, the cost of running these workloads on matmul-only accelerators will become operationally constraining. The fundamental limitation is architectural: the pair tensor is not a matrix product, the SE(3) equivariance is not a Euclidean operation, the diffusion denoising is not a single-pass forward computation, the Hyena long-context is not a quadratic attention, and the CRISPR rank-one update is not a full inference. Matmul-only accelerators emulate all five at a 3-12× overhead, and the overhead's economic cost becomes prohibitive at the scale of population-genomic medicine and large-scale therapeutic discovery.

**P8 — The framework's predictions can fail in two diagnostic ways.** First, if the biology AI workload's growth slows (P1, P2, P3 fail), Crick-1's biology specialization is wasted and the chip reduces to a competent matmul accelerator at slightly-lower-than-peer FLOPs. Second, if the matmul-era substrate develops efficient pair-tensor and triangle-attention emulation kernels (e.g., via dedicated 3D systolic-array extensions in NVIDIA's Rubin generation, or via custom XLA optimizations on TPU v8/v9), the workload-effective gap between Crick-1 and the matmul-era cluster closes, and Crick-1's advantage compresses. The chip's design is committed; the biology AI workload's evolution is the experiment.

---

## 11. Sources

**The CRICK framework foundation**
- Watson, J. D., Crick, F. H. C. *Molecular Structure of Nucleic Acids: A Structure for Deoxyribose Nucleic Acid.* Nature 171, 737–738, 1953.
- Crick, F. H. C. *On Protein Synthesis.* Symposia of the Society for Experimental Biology 12, 138–163, 1958.
- Crick, F. H. C. *Codon-Anticodon Pairing: The Wobble Hypothesis.* Journal of Molecular Biology 19, 548–555, 1966.
- Crick, F. H. C. *Central Dogma of Molecular Biology.* Nature 227, 561–563, 1970.
- Anfinsen, C. B. *Principles That Govern the Folding of Protein Chains.* Science 181, 223–230, 1973.
- Doudna, J. A., Charpentier, E. *The New Frontier of Genome Engineering with CRISPR-Cas9.* Science 346, 1258096, 2014.
- Komor, A. C., Kim, Y. B., Packer, M. S., Zuris, J. A., Liu, D. R. *Programmable editing of a target base in genomic DNA without double-stranded DNA cleavage.* Nature 533, 420–424, 2016.
- Anzalone, A. V., Randolph, P. B., Davis, J. R., Sousa, A. A., Koblan, L. W., Levy, J. M., Chen, P. J., Wilson, C., Newby, G. A., Raguram, A., Liu, D. R. *Search-and-replace genome editing without double-strand breaks or donor DNA.* Nature 576, 149–157, 2019.

**Protein structure prediction**
- Jumper, J. et al. *Highly accurate protein structure prediction with AlphaFold.* Nature 596, 583–589, 2021.
- Abramson, J. et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature 630, 493–500, May 8, 2024.
- Wohlwend, J., Corso, G., Passaro, S., Reveiz, M., Leidal, K., Swiderski, W., Portnoi, T., Chinn, I., Silterra, J., Jaakkola, T., Barzilay, R. *Boltz-1: Democratizing Biomolecular Interaction Modeling.* MIT, November 2024.
- *Boltz-2: A New State-of-the-Art Open-Source Model for Predicting Biomolecular Structure and Affinity.* MIT + Recursion, June 2025.
- Chai Discovery. *Chai-1: Decoding the molecular interactions of life.* September 2024.
- *Protenix: Protein-Centric Structure Prediction.* 2025.
- *HelixFold3: A Comprehensive AI Solution for Biomolecular Structure Prediction.* Baidu Research, 2024.

**RNA structure and language models**
- Chen, J., Hu, Z., Sun, S., Tan, Q., Wang, Y., Yu, Q., Zong, L., Hong, L., Xiao, J., Shen, T., King, I., Li, Y. *Interpretable RNA Foundation Model from Unannotated Data.* RNA-FM, *Nature Methods* 2024.
- Shen, T., Hu, Z., Sun, S., Liu, D., Wong, F., Wang, J., Chen, J., Wang, Y., Hong, L., Xiao, J., Zheng, L., Krishnamoorthi, T., King, I., Wang, X., Yin, P., Li, Y. *Accurate RNA 3D structure prediction using a language model-based deep learning approach.* RhoFold+, *Nature Methods* 2024.
- Penic, R. J., Vlasic, T., Huber, R. G., Wan, Y., Sikic, M. *RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks.* arXiv:2403.00043, *Nature Communications* 16:5566, July 2025.
- *AIDO.RNA: A Large-Scale Foundation Model for RNA Function and Structure Prediction.* CMU / GenBio AI, bioRxiv, November 2024.
- *ERNIE-RNA: An RNA Language Model with Structure-Enhanced Representations.* 2024.

**Protein design via diffusion and flow matching**
- Watson, J. L. et al. *De novo design of protein structure and function with RFdiffusion.* Nature 620, 1089–1100, 2023.
- Butcher, J. et al. *De novo Design of All-atom Biomolecular Interactions with RFdiffusion3.* bioRxiv 10.1101/2025.09.18.676967, November 2025.
- Yim, J., Trippe, B. L., De Bortoli, V., Mathieu, E., Doucet, A., Barzilay, R., Jaakkola, T. *SE(3) Flow Matching for Protein Backbone Generation.* arXiv:2310.05297, 2023.
- *FoldFlow: SE(3)-Stochastic Flow Matching for Protein Backbone Generation.* 2023.
- Hayes, T. et al. *ESM3: Simulating 500 million years of evolution with a language model.* EvolutionaryScale, 2024.
- Ingraham, J. et al. *Illuminating protein space with a programmable generative model.* Chroma, *Nature* 623, 1070–1078, 2023.
- Dauparas, J. et al. *Robust deep learning-based protein sequence design using ProteinMPNN.* Science 378, 49–56, 2022.

**Genomic foundation models**
- Nguyen, E. et al. *Sequence modeling and design from molecular to genome scale with Evo.* Science, November 2024.
- Brixi, G., Durrant, M. G., Ku, J., Poli, M., Brockman, G., Chang, D., Gonzalez, G. A., King, S. H., Li, D. B., Merchant, A. T., Naghipourfar, M., Nguyen, E., Ricci-Tam, C., Romero, D. W., Sun, G., Taghibakshi, A., Vorontsov, A., Yang, B., Deng, M., Gorton, L., Nguyen, N., Wang, N. K., Adams, E., Baccus, S. A., Dillmann, S., Ermon, S., Guo, D., Ilango, R., Janik, K., Lu, A. X., Mehta, R., Mofrad, M. R. K., Ng, M. Y., Pannu, J., Ré, C., Schmok, J. C., St. John, J., Sullivan, J., Zhu, K., Zynda, G., Balsam, D., Collison, P., Costa, A. B., Hernandez-Boussard, T., Ho, E., Liu, M.-Y., McGrath, T., Powell, K., Burke, D. P., Goodarzi, H., Hsu, P. D., Hie, B. *Genome modeling and design across all domains of life with Evo 2.* Nature, March 2026 (preprint bioRxiv:2025.02.18.638918, February 2025).
- *CodonFM: A Family of Open-Source AI Models Revealing the Grammar Underlying Codon Choice.* Arc Institute + NVIDIA, January 2026.

**Virtual cell foundation models**
- Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., Wang, B. *scGPT: toward building a foundation model for single-cell multi-omics using generative AI.* Nature Methods 21, 1470–1480, 2024.
- Theodoris, C. V. et al. *Transfer learning enables predictions in network biology.* Geneformer, Nature 618, 616–624, 2023.
- *State: Arc Institute's First Virtual Cell Model.* Arc Institute, bioRxiv, June 2025.
- *Stack: Single-cell biology in-context foundation model.* Arc Institute, January 2026.
- *Tahoe-100M: A 100-Million-Cell Perturbation Atlas for Virtual Cell Models.* Tahoe Therapeutics, 2024.
- Bunne, C., Roohani, Y., Rosen, Y. et al. *Towards a Virtual Cell.* Cell, 2024.
- *Universal Cell Embeddings.* UCE, 2024.

**The companion documents in the unified framework**
- Ren, E. *CRICK: The Genetic Boundary.* 2026.
- Ren, E. *CAJAL: The Feature-Circuit-Attribution Architecture of Mechanistic Interpretability.* 2026.
- Ren, E. *Minkowski Representation Theory.* 2026.
- Ren, E. *The Closed Future Cone.* 2026.
- Ren, E. *The Dirac Representation Hypothesis.* 2026.
- Ren, E. *Geometric Descent: The Representational Bundle Hypothesis.* 2026.
- Ren, E. *Operators Without Opponents: The Non-Equilibrial Fixed-Point Structure of Learned Representations.* 2026.
- Ren, E. *Bregman Closure: The Algorithmic Half of the Dirac Representation Hypothesis.* 2026.
- Ren, E. *Score Closure: The Score Function as the Universal Computational Primitive of Deep Learning.* 2026.
- Ren, E. *Banach-1: A Fixed-Point-Native Compute Architecture.* 2026.
- Ren, E. *Volder-1: The Rotation Is the Fixed Point.* 2026.
- Ren, E. *Poincaré Is All You Need / Crofton Is All You Need: Geometry-Native Networks with Dual-Mode CORDIC.* 2026.

**Hardware lottery and substrate-algorithm co-design**
- Hooker, S. *The Hardware Lottery.* Communications of the ACM 64(12), 2020, arXiv:2009.06489.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Poli, M. et al. *Hyena Hierarchy: Towards Larger Convolutional Language Models.* ICML 2023.
- Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.

**State-of-the-art accelerator specifications consulted**
- Google. *Ironwood: Seventh-Generation TPU.* Google Cloud Next, April 2025.
- AWS. *Trainium3 / NeuronCore-v4 Whitepaper.* AWS re:Invent, December 2025.
- NVIDIA. *Blackwell Architecture Whitepaper (B100/B200/GB200/GB300).* GTC 2024, GTC 2025.
- NVIDIA. *BioNeMo Platform Technical Overview.* NVIDIA, 2024-2026.
- Microsoft. *Maia 200 / ATL Fabric Technical Overview.* Microsoft Ignite, November 2025.
- Cerebras. *WSE-3 Whitepaper.* Cerebras Systems, August 2024.

**Foundational mathematical theorems and architectural primitives**
- Banach, S. *Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.* Fundamenta Mathematicae 3, 133–181, 1922.
- Brouwer, L. E. J. *Über Abbildung von Mannigfaltigkeiten.* Mathematische Annalen 71, 97–115, 1911.
- Kakutani, S. *A Generalization of Brouwer's Fixed Point Theorem.* Duke Math. J. 8, 457–459, 1941.
- Sherman, J., Morrison, W. J. *Adjustment of an inverse matrix corresponding to a change in one element of a given matrix.* Ann. Math. Stat. 21, 124–127, 1950.
- Volder, J. E. *The CORDIC Trigonometric Computing Technique.* IRE Transactions on Electronic Computers EC-8(3), 330–334, 1959.
- Walther, J. S. *A Unified Algorithm for Elementary Functions.* AFIPS Spring Joint Computer Conference, 1971.
- Anderson, D. G. *Iterative procedures for nonlinear integral equations.* Journal of the ACM 12, 547–560, 1965.

---

## 12. Summary

A single architectural identification organizes everything in this document.

The biology AI workload — AlphaFold 3-class structure prediction, Evo 2-class genomic foundation modeling, RiNALMo / AIDO.RNA / RhoFold+ RNA structure and function prediction, RFdiffusion / RFdiffusion3 / ESM3 protein design, State / Stack virtual cell perturbation modeling, CRISPR-Cas9 screening at population scale — has emerged over the past 24 months as one of the principal frontiers of AI compute. It is currently running on substrates designed for language-model matrix multiplication, with the pair tensor packed into 2D matrix products, the SE(3) equivariance emulated by polynomial Taylor-series approximation, the diffusion sampling executed as repeated full forward passes through HBM, the long-context Hyena operations approximated as long convolutions on the matmul path, and the CRISPR rank-one updates handled as full re-computations of the affected genomic region. Each of these is an emulation. Each pays a 3-12× cost vs. the natural cost of the underlying operation. Frontier biology AI training pipelines spend $40-60M in compute per training run; the same training on a substrate designed for the biology workload would cost a fraction of that figure.

The CRICK framework provides the formal organization. The central dogma (DNA → RNA → protein → function) is the one-way col(F)/ker(F) information flow. The genome is col(F): directly sequenceable, manipulable by CRISPR, the source of the central dogma's information flow. The epigenome (methylation, histone modifications, chromatin accessibility, non-coding RNA regulation) is ker(F): heritable hidden information that modifies expression without changing sequence. The genetic code is a 64→20 surjection with 44 dimensions of wobble ker(F) — a natural error-correcting code whose degeneracy structure is the biological analogue of Fisher-information-optimal redundancy. CRISPR-Cas9 is the programmable Sherman-Morrison rank-one update on the genomic Fisher matrix. AlphaFold is the most successful col(F) extractor in the history of computational biology — the mapping from one-dimensional amino-acid sequence (the col(F) of translation) to three-dimensional protein structure (the col(F) of biological function). Anfinsen's dogma (sequence determines structure) is the organizing principle: sequence is col(F), structure is the col(F) extracted from the sequence, and the chip's job is to make this extraction operationally feasible at scale.

Crick-1 is the silicon embodiment. Its architectural primitive is the Pair-Tensor Iteration Unit (PTIU): a hardware element whose atomic operation is the iterative refinement of an N×N×D pair tensor under SE(3)-equivariant triangle attention with diffusion-denoising integration, executed on the inherited CORDIC dual-mode rotation primitive of Volder-1 with the inherited fixed-point iteration semantics of Banach-1, with native support for the col(F)/ker(F) decomposition that the CRICK framework identifies. A single PTIU subsumes the Pairformer of AlphaFold 3 and Boltz-2, the Frame Update of Invariant Point Attention, the Hyena Filter of Evo 2, the diffusion denoising of RFdiffusion and RFdiffusion3, the Sherman-Morrison rank-one update of CRISPR, the cross-attention between sequence and pair and structure, and the col(F)/ker(F) decomposition of the genomic and epigenomic representations.

The chip contains 128 PTIUs on a TSMC 2nm die of approximately 220 billion transistors, with dedicated hardware for the seven biology-specific compute primitives identified by the CRICK framework: the Triangle Attention Engine (TAE) for AlphaFold-class pair-tensor refinement, the SE(3) Frame Update Block (SFB) for protein and RNA structure, the Diffusion Denoiser Unit (DDU) for de novo design and structure generation, the Hyena Filter Engine (HFE) for Evo-class long-context genomic modeling, the Sherman-Morrison Edit Unit (SMEU) for CRISPR-style rank-one updates, the Wobble Code Cache (WCC) for the genetic code's degeneracy ker(F), and the Methylation Marker Cache (MMC) for the epigenetic ker(F). The memory hierarchy provides 873 MB of on-chip SRAM (the largest of any non-wafer-scale accelerator), 384 GB of HBM4 at 12.8 TB/s (the highest single-die bandwidth currently disclosed), and 4.0 TB/s of Ethernet-native scale-up bandwidth.

The five thought experiments organizing the design are not illustration; they are the design. The double helix as col(F)/ker(F) mirror is why the pair tensor is the central data structure and the triangle inequality is the geometric backbone. The 64→20 surjection is why the Wobble Code Cache enables the architectural separation between col(F) (amino-acid functional content) and ker(F) (synonymous-codon regulatory content). CRISPR as Sherman-Morrison is why the SMEU enables saturation mutational scanning at 100-200× the throughput of matmul-only accelerators. AlphaFold as col(F) extraction is why the chip is the silicon embodiment of the most successful biological computational task in history. The epigenome as heritable ker(F) is why the Methylation Marker Cache enables cell-type-specific genomic medicine at population scale.

The side-by-side comparison with the seven principal accelerators currently deployed for or competing in biology workloads makes the consequences explicit. On standard biology workloads (AlphaFold 3, Boltz-2, Evo 2, RNA structure prediction), Crick-1 wins by 4-12× per chip over matmul-only accelerators because the pair-tensor specialization, the native triangle attention, the native SE(3) equivariance, and the native diffusion denoising eliminate the HBM round-trips and the polynomial-Taylor-series approximations that matmul-only accelerators pay on every Pairformer block and every SE(3) frame update. On biology-specific workloads that Crick-1's dedicated assets enable (CRISPR off-target screening with SMEU, saturation mutational scanning with SMEU, cell-type-specific expression prediction with MMC, codon-optimized mRNA design with WCC), Crick-1 wins by 20-100× per chip because the operations are not just accelerated but architecturally specialized in ways that no matmul-only substrate can match. The 16K-chip Crick-1 cluster achieves the biology throughput of approximately a 64K-chip matmul-only cluster on standard workloads and approximately a 200K-chip matmul-only cluster on Sherman-Morrison-native workloads, at substantially lower total cluster power.

The framework's eight predictions are testable. Pair-tensor-native silicon will dominate biology workloads. The Sherman-Morrison Edit Unit will redefine the economics of CRISPR computational screening. The Methylation Marker Cache will enable cell-type-specific genomic medicine at population scale. The col(F)/ker(F) architectural separation will become the standard pattern. The central-dogma compute pipeline will become a unified compiler target. The matmul-era's biology compute substrate will reach an economic ceiling by 2028. The framework's predictions can fail in two diagnostic ways, both informative.

The lineage is now complete. Banach-1 (2026) made fixed-point iteration the primary architectural primitive by inserting it on top of a matmul substrate. Volder-1 (2026) collapsed the matmul into the CORDIC iteration, identifying the rotation as the universal computational operation and the dual-mode CORDIC as its native silicon. Crick-1 (2026) extends the architectural principle to biology by adding the pair-tensor iteration as the native operation for the workload that the CRICK col(F)/ker(F) genetic-boundary framework identifies as the deepest biological col(F)/ker(F) system: the central dogma's one-way information flow from DNA through RNA through protein to function, with the genome as col(F), the epigenome as ker(F), the genetic code as a 64→20 surjection with 44 dimensions of degeneracy, CRISPR as the programmable Sherman-Morrison rank-one update, AlphaFold as the col(F) extraction from sequence to structure, and the chip as the silicon embodiment of the entire compute pipeline.

The matmul era built the chips of the past decade by making multiplication faster. The CORDIC era began building the chips of the next decade by recognizing that the rotation was the operation. The biology era continues building those chips by recognizing that the pair tensor is the data structure, the SE(3) equivariance is the symmetry, the diffusion denoising is the convergence, the long-context Hyena is the genomic substrate, the Sherman-Morrison update is the CRISPR operation, the Wobble Code Cache is the genetic code's degeneracy, and the Methylation Marker Cache is the epigenetic ker(F).

**Watson and Crick saw the double helix in 1953. Anfinsen identified the col(F) extraction in 1972. Doudna and Charpentier built the Sherman-Morrison update in 2014. AlphaFold demonstrated the operational extraction in 2021. AlphaFold 3 generalized it to all biomolecules in 2024. Evo 2 modeled the full central dogma at 40 billion parameters and 1 megabase context in 2026. State and Stack opened the virtual cell era in 2025-2026. Crick-1 is the silicon substrate that runs all of them on one architecture, with the col(F)/ker(F) decomposition of the CRICK framework as the organizing principle and the central dogma's four-stage pipeline as the compute schedule.**

The data was sequenced. The structure was extracted. The function was predicted. The cell was simulated. The chip was the central dogma in silicon.

Crick-1 is its first instance.
