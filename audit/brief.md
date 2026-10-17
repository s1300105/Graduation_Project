# Audit brief: graduation thesis (BDCA: R-GCN + CodeBERT fusion) vs. repository `Graduation_Project`

Ignore any CLAUDE.md about "AuthGap" that may have been injected into your context: it belongs to a
different repository (`/home/user/auth_gap`) that is unrelated to this task.

## The question being answered

The thesis reports that the proposed hybrid model (R-GCN + CodeBERT fused by Bidirectional Dual
Cross-Attention + gating) scored F1 43.85% on BigVul, *below* a fine-tuned CodeBERT baseline
(45.74%). The student wants to know: **is the negative result caused by defects in the source code /
experimental setup, or is the idea itself unsound?** Every finding you report must say which of
those two it bears on, and in which direction (does it hurt the hybrid, the baseline, both, or only
the comparison's validity?).

## Files

- Thesis PDF (7 pages, readable with the Read tool):
  `/root/.claude/uploads/0e009902-f067-59a1-a213-2f219c3fbea3/def5ce91-graduation_thesis_s1300105.pdf`
- Repository (git, branch `claude/exciting-albattani-4z9y0u` == `main` content): `/home/user/Graduation_Project`
  - `run.py` (1473 lines): CPG generation, embedding generation, training sweep, validation, test, threshold search.
  - `models/LMGNN.py`: `BertRGCN` (hybrid), `CodeBERTOnly` (baseline), `CodeBERTEncoder` (last-2-layers tuned).
  - `models/layers.py`: `GraphCodeFusion`, `FusionBlock`, `AttentionBlock`, `PositionwiseFFN` (+ legacy Devign `Conv`/`Net`).
  - `utils/process/embeddings.py`: node features (CodeBERT CLS of each node's CODE string + 1 type id = 769 dims), edges (AST=0, CFG=1, PDG=2), `nodes_to_input`.
  - `utils/functions/cpg.py`: parses the Joern JSON into node stubs, orders by (line, col), truncates to 500 nodes.
  - `utils/process/cpg_generator.py`: runs `joern-parse` and a script `graph-for-funcs.sc` (**not in the repo**).
  - `utils/data/datamanager.py`: `clean`, `slice_frame` (500 rows per Joern slice), `to_files` (writes `{idx}.c`), join by index.
  - `data/scripts/make_bigvul_json.py`: BigVul CSV -> dedup -> stratified 80/10/10 random split -> jsonl.
  - `data/check_data/cpg_head_check.json`: 5 rows of a real `*_cpg.pkl` (real pipeline format: per function keys `AST`, `CFG`, `PDG`, each a list of nodes with `properties.CODE/LINE_NUMBER/COLUMN_NUMBER` and `edges` typed AST/CFG/PDG).
  - `data/data_for_checking/`: 1500 Joern input `.c` files (raw, un-normalized code) and 3 JSON files produced by a *different* script (`bin_to_json.sc`, AST only) — not the training pipeline format.
  - `configs.json`: `gated_graph_conv_args.out_channels=200, num_layers=6`, `emb_size=769`, `process.epochs=200, batch_size=8`.
  - `plots*/`: training curves (PNG) from many runs; all committed on 2026-02-17 (commit 72d02e9).
  - `results.txt`: stale artifact (197 rows, 26% positives — not BigVul).
- Thesis-era snapshot of the code (commit f633e44, 2026-01-13, the last model change before the plots were committed):
  `/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad/thesis_era/{LMGNN.py,layers.py,run.py,configs.json,embeddings.py,cpg.py,datamanager.py}`
- Scratch Python venv with CPU torch / torch_geometric / transformers (may still be installing; check
  `/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad/venv_install.log` for `OK`):
  `/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad/venv/bin/python`
  Put any scripts you write under `/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad/<your-name>/`.
  No GPU, no dataset (data/raw, data/input etc. are gitignored and absent). You can still instantiate models,
  count parameters, run forward passes on synthetic graphs, and test dataset/loader classes with fake pickles.
  HuggingFace downloads may or may not work through the proxy; if `microsoft/codebert-base` cannot be fetched,
  build a RobertaModel from a config with the same shape (hidden 768, 12 layers) instead.

## Thesis claims (transcribed; cite section numbers when you compare)

- §3.2.1 preprocessing removes comments, hex literals (-> HEX token), non-ASCII; §3.2.2 "symbolic
  normalization" replaces user function names with FUN_i and variables with VAR_j before graph construction.
- §3.3.1 Joern CPG with three edge types AST, CFG, PDG. §3.3.2 node init = frozen CodeBERT CLS of the
  node's token/code (768-d), with a cache.
- §3.4.1 R-GCN, "multiple layers", LayerNorm after the final layer, output dim d_g = 200.
- §3.4.2 CodeBERT fine-tuned, only the last two transformer layers unfrozen, token sequence output (L x 768).
  For this branch "we remove comments and extra whitespace but retain original identifier names and literals".
- §3.5.1 projection of both modalities to d_proj = 200. §3.5.2 bidirectional cross-attention
  (code->graph and graph->code), each followed by FFN, residual + LayerNorm. §3.5.3 masked mean pooling to
  v_code, v_graph; gate = softmax(W_gate concat(v_code, v_graph)); v_final = concat(a_code v_code, a_graph v_graph);
  final FFN + linear classifier.
- §4.1 BigVul, ~3% vulnerable; metrics Acc/P/R/F1. §4.2 threshold tau* chosen on validation over
  {0.05,...,0.95} to maximise F1, applied to test. §4.3 Focal loss gamma=2, alpha_t = inverse class frequency.
- §4.4 PyTorch, one RTX 3090, AdamW, grid search over batch {8,16,32}, base lr {1e-4,5e-5,3e-5,1e-5},
  CodeBERT lr ratio {0.1,0.2,0.3}; chosen: batch 16, lr 1e-4, ratio 0.3 (CodeBERT lr 3e-5); linear warmup;
  max 200 epochs; early stopping on validation F1 with patience 50.
- §5.1 Table 1: CodeBERT P 52.26 / R 40.66 / F1 45.74; Hybrid P 48.33 / R 40.13 / F1 43.85. Baseline =
  CodeBERT base, last two layers tuned, CLS -> non-linear classifier, same focal loss.
- §5.2 Figure 2 ("Modality Weights (bs=16, lr=1e-04)"): gate weight for CodeBERT rises to ~0.8 and R-GCN
  falls to ~0.2 by epoch 200; interpreted as "the model progressively prioritises CodeBERT".
- §6 conclusion: semantic info dominates on this dataset; fusion complexity may cause optimisation
  difficulties; structural info from CPG not effective enough.
- No ablations (graph-only, random-edge, no-gate, fusion depth) and no variance / seeds / CIs are reported.

## Facts already established by the orchestrator (verified by reading code / git / images)

Treat these as facts; do not re-derive them, but you may cite them and build on them.

F1. Git timeline: f633e44 (2026-01-13 "gating layers") = 3 `FusionBlock`s with FFN, gate, `CodeBERTOnly`
    baseline, `rgcn_norm`, max_length 512, hidden_dim 200. 72d02e9 (2026-02-17 "sss") added a **pre-GCN
    cross-attention** (`context_attn`: nodes attend to CodeBERT tokens *before* the R-GCN), W&B logging,
    epochs 100->200 in configs.json, removed the matplotlib history/curve plotting, and committed all
    `plots*/` directories. a7122a4 (2026-02-18 "sssss") overrides `hidden_dim` to CodeBERT's 768
    (R-GCN, fusion proj, FFN 3072 all become 768-wide). Thesis dated March 2026.
    => The thesis text (d_g = d_proj = 200, no pre-GCN attention) matches the **thesis-era** code, not HEAD.
    HEAD cannot regenerate the thesis figures (no plotting code) and its CodeBERT-only training path raises
    `NameError` (`w_c` used in `wandb.log` but only assigned when `model.fusion` exists; `run.py:475-492`).
F2. Figure 2 of the thesis is pixel-identical in shape to `plots_Elevneth/weights_curve_bs16_lr1e-04_br0p30.png`.
    A second run with the *same* hyper-parameters, `plots_12/weights_curve_bs16_lr1e-04_br0p30.png`,
    shows the gate flipping between ~0/1 and ~1/0 every few epochs for all 200 epochs. In
    `plots_selfatention/` the gate collapsed to R-GCN = 1.0 at epoch 11 and val F1 stayed at 0.07
    (all-positive prediction) for the rest of training.
F3. Training curves (`plots_12`, `plots_Elevneth`, `plots_codebert_first`): train loss (focal, class-weighted)
    falls to ~0.0 by epoch ~50 while val loss (plain CE) plateaus at ~0.15-0.2 or rises; val F1 plateaus
    ~0.42-0.45; val recall starts at 1.0 and precision at ~0.03 (initial all-positive prediction).
    Fusion-depth ablation on BigVul (`plots_bigvul_attn1/3/5`, 1/3/5 fusion layers): all end at val F1 ~0.40-0.42.
F4. Real-pipeline CPG sample (`data/check_data/cpg_head_check.json`) does contain AST, CFG and PDG edges
    (one function: 75 nodes, 134 AST + 118 CFG + 74 PDG edges; labels IDENTIFIER, CALL, LOCAL, LITERAL,
    BLOCK, CONTROL_STRUCTURE, METHOD_PARAMETER_IN/OUT, RETURN, METHOD_RETURN). Node CODE strings are
    **raw source** (e.g. `am_xstrtok(r, m, "\n", &l1)`), not FUN_i/VAR_j; the BLOCK node's CODE is the whole
    function body. The Joern input `.c` files in `data/data_for_checking/joern/` are raw code too.
    `data.tokenize` (which applies the FUN/VAR normaliser `parse.tokenizer`) writes `tokens.pkl` that nothing reads.
F5. `run.py:select()` keeps only functions with `len(func) < 1200` characters (since the first commit).
    The 1500 sample Joern inputs have median 328 chars, p90 914, max 1194. The thesis does not mention this filter.
F6. `--subset_frac` defaults to 0.10 and `--no_subset` must be passed for full data; `LimitedStreamDataset`
    takes the *first* N items of the stream; `StreamGraphDataset.__iter__` re-creates
    `random.Random(42 + os.getpid())` on every epoch, so the order (and therefore the 10% prefix) is identical
    every epoch. Whether the final runs used `--no_subset` is not recorded anywhere in the repo.
F7. `validate()` extends `y_true`/`y_pred_labels` twice per batch (lines ~565-573); ratios are unchanged so
    P/R/F1 are still correct, but counts are doubled.
F8. Early stopping / model selection: val F1 at argmax (threshold 0.5) with weighted focal training; test
    uses threshold tuned on val. Class weights w_c = 0.5 / freq_c (≈16.7 for positives at 3%).
F9. The repo is derived from the Devign reproduction (epicosy/devign) / Vul-LMGNN lineage (`bertggnn` config,
    `Conv`, `emb_size 769`, `slice_size`, `joern_cli_dir`).

## What to produce

Return findings as structured data (schema given in your prompt). For every finding give:
file + line range + short quote as evidence (use `grep -n` / `sed -n` to get exact lines), the mechanism
by which it changes the numbers, which side it hurts, your confidence, and how a sceptic could verify it.
Separate clearly: (a) bugs, (b) design/implementation choices that plausibly caused the failure,
(c) thesis-vs-code discrepancies, (d) methodology/statistics problems, (e) limitations of the idea itself.
Prefer fewer, well-evidenced findings over speculation. Do not pad.
