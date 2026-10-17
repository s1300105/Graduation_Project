export const meta = {
  name: 'thesis-code-audit',
  description: 'Audit graduation thesis (R-GCN+CodeBERT fusion) against Graduation_Project repo: find defects, adversarially verify, decide code-vs-idea',
  phases: [
    { title: 'Find', detail: '7 independent finders over code, data, plots, literature, provenance' },
    { title: 'Merge', detail: 'dedup/merge overlapping findings' },
    { title: 'Verify', detail: '3 refuters per finding (code trace / impact / empirical)' },
    { title: 'Critic', detail: 'completeness critic + verify its additions' },
  ],
}

const BRIEF = '/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad/brief.md'
const SCRATCH = '/tmp/claude-0/-home-user/0e009902-f067-59a1-a213-2f219c3fbea3/scratchpad'
const REPO = '/home/user/Graduation_Project'
const PDF = '/root/.claude/uploads/0e009902-f067-59a1-a213-2f219c3fbea3/def5ce91-graduation_thesis_s1300105.pdf'

const FINDING = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    title: { type: 'string' },
    category: { type: 'string', enum: ['bug', 'design_choice', 'thesis_code_mismatch', 'methodology', 'idea_limitation', 'reproducibility'] },
    severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
    claim: { type: 'string' },
    evidence: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'string' }, quote: { type: 'string' } }, required: ['file', 'quote'] } },
    mechanism: { type: 'string' },
    hurts: { type: 'array', items: { type: 'string', enum: ['hybrid', 'baseline', 'both', 'comparison_validity', 'thesis_text', 'reproducibility'] } },
    bears_on: { type: 'string', enum: ['code', 'idea', 'both', 'neither'] },
    confidence: { type: 'number' },
    how_to_verify: { type: 'string' },
    suggested_fix: { type: 'string' },
  },
  required: ['id', 'title', 'category', 'severity', 'claim', 'evidence', 'mechanism', 'hurts', 'bears_on', 'confidence', 'how_to_verify'],
}
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: { finder: { type: 'string' }, summary: { type: 'string' }, findings: { type: 'array', items: FINDING } },
  required: ['finder', 'summary', 'findings'],
}
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    confidence: { type: 'number' },
    reasoning: { type: 'string' },
    evidence: { type: 'string' },
    corrected_claim: { type: 'string' },
    severity_after: { type: 'string', enum: ['critical', 'major', 'minor', 'none'] },
  },
  required: ['refuted', 'confidence', 'reasoning', 'evidence', 'severity_after'],
}
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string' },
    code_vs_idea: { type: 'string', enum: ['mostly_code', 'mostly_idea', 'both_substantially', 'cannot_tell_from_evidence'] },
    reasoning: { type: 'string' },
    contradictions: { type: 'array', items: { type: 'string' } },
    ranked_finding_ids: { type: 'array', items: { type: 'string' } },
    missing: { type: 'array', items: FINDING },
  },
  required: ['verdict', 'code_vs_idea', 'reasoning', 'contradictions', 'ranked_finding_ids', 'missing'],
}

const COMMON = `Read ${BRIEF} first (it holds file paths, the thesis claims, and facts already verified; do not re-derive the facts, cite them). Work in ${REPO}. Use grep -n / sed -n so every evidence item carries exact line numbers and a short verbatim quote. When a point depends on which code version produced the thesis, say so explicitly: thesis-era snapshot is ${SCRATCH}/thesis_era/, HEAD is the repo. Report only findings you can evidence; give each an id prefixed with your finder key. Quality over quantity. Your final answer is the structured output only.`

const FINDERS = [
  { key: 'trainloop', prompt: `${COMMON}
Finder key: trainloop. Scope: the training / validation / test / threshold code in ${REPO}/run.py (functions FocalLoss, StreamGraphDataset, LimitedStreamDataset, BalancedStreamDataset (unused?), make_stream_loader_for_split, train, validate, test, _collect_probs_and_labels, search_best_threshold, evaluate_with_threshold, _count_targets_in_files, and the __main__ sweep/test blocks) and the same functions in the thesis-era snapshot. Find defects or choices that change the reported numbers or the hybrid-vs-baseline comparison: optimiser param grouping (which params get bert_lr vs base_lr; weight decay 5e-4 on LayerNorm/bias?), scheduler total_steps vs early stopping (how much of the schedule is actually run; warmup length in epochs), class weights x focal loss interaction and the initial all-positive regime, early-stopping criterion vs the threshold-tuned test metric, OOM-skip swallowing batches, subset/same-order-every-epoch (fact F6: quantify how much data diversity is lost and whether it affects hybrid and baseline differently), evaluation-time batch size from configs.json, anything in test mode that could make Table 1 numbers incomparable between model types, and anything else. For each, state whether it also applies to the baseline run (both models go through the same loop) so the reader knows whether it biases the comparison or only lowers both.` },
  { key: 'model', prompt: `${COMMON}
Finder key: model. Scope: ${REPO}/models/LMGNN.py and ${REPO}/models/layers.py versus thesis sections 3.3-3.5 and 4.4, for BOTH the thesis-era snapshot and HEAD. Produce (1) a claim-by-claim reconciliation: for each architectural statement in the thesis say matches / differs / not in thesis, with line evidence (e.g. d_g, d_proj, number of R-GCN layers, residuals or dropout inside the R-GCN stack, num_bases, LayerNorm placement, number of fusion layers, FFN size, pooling, gate formulation, classifier, which CodeBERT layers are trainable, pooler, max_length, HEAD-only pre-GCN context attention and the 768 override). (2) Optimisation pathologies implied by the implementation, each argued concretely from the code: e.g. 6 stacked RGCNConv + ReLU with no residual/dropout (over-smoothing / signal attenuation into mean pooling), randomly-initialised 3-layer bidirectional cross-attention + FFN stacks trained from scratch on a small 3%-positive set (capacity vs data), gate = softmax over a bare linear layer with no temperature/regularisation (fact F2: oscillation / collapse), LayerNorm after the gated concat partially undoing the gate scale, the graph branch being seeded with CodeBERT CLS per node (how much of the graph branch is really text), input feature scale (type_id 0/1 next to un-normalised CLS vectors). (3) Use the scratch venv to instantiate CodeBERTOnly and BertRGCN (thesis-era file and HEAD file; import them by path, pass the configs.json args, Conv from layers.py) and report trainable / total parameter counts for: baseline, thesis-era hybrid (200-d), HEAD hybrid (768-d); and run one forward pass on a synthetic batch of 2 small graphs to confirm shapes. Classify each finding as bug vs design choice, and say whether it bears on code or idea.` },
  { key: 'graph', prompt: `${COMMON}
Finder key: graph. Scope: how the graph that the R-GCN sees is built: ${REPO}/utils/functions/cpg.py, ${REPO}/utils/process/embeddings.py, ${REPO}/utils/process/cpg_generator.py, ${REPO}/utils/data/datamanager.py, the CPG_generator/Embed_generator functions in run.py, and the real-pipeline sample ${REPO}/data/check_data/cpg_head_check.json (5 JSON rows; parse it with python3 -I). Questions: Are the AST/CFG/PDG edges correctly mapped to edge_type 0/1/2 and directions preserved (trace _attach_edge/build_edges_all including the node_in/node_out naming)? Are PDG edges in Joern's JSON actually data-dependence edges or CDG too? How many nodes have empty CODE, and what CLS do they get? What fraction of nodes are IDENTIFIER/LITERAL whose CodeBERT CLS of a single token carries little? The BLOCK node whose CODE is the entire function body (fact F4): what does that imply for structure-vs-semantics separation? Node ordering by (line, col) and the 500-node cut: which nodes get dropped first? The fp16 CodeBERT node-embedding path and cache keyed by code string (identical strings give identical vectors regardless of context). filter_nodes(strict=False) keeping nodes without line numbers. Thesis section 3.2.2 symbolic normalisation vs fact F4 (raw code in nodes): confirm by tracing where tokens.pkl is consumed (grep for FILES.tokens and 'tokens'). Also check cpg_generator.joern_create/json_process for silent data loss (timeouts, functions skipped, Index join dropping rows) and whether the thesis-era embeddings.py differs. Use the venv to run parse_to_functions + nodes_to_input on the 5 sample rows (mode='fast' to avoid CodeBERT, or 'codebert' if quick) and report per-graph node/edge counts by type, self-loop fallbacks, and any exception. Say for each finding whether it weakens the structural signal (hurting the hybrid's chance) and whether it is a bug or a design choice.` },
  { key: 'data', prompt: `${COMMON}
Finder key: data. Scope: dataset construction and experimental protocol: ${REPO}/data/scripts/make_bigvul_json.py, select()/clean()/shuffle in run.py, split handling, the sweep protocol in run.py __main__, thesis sections 4-5. Items to evidence: the undisclosed len(func) < 1200 filter (fact F5) - what kind of functions it removes (reason from BigVul's known length distribution; cite literature numbers if you can find them with WebSearch, e.g. LineVul/BigVul statistics; state clearly when a number is unverified), and whether vulnerable functions are longer on average so the filter changes the positive rate; dedup by exact string vs near-duplicates; random function-level stratified split on BigVul (leakage across commits/projects; what the literature says about BigVul splits); whether the default 10% subset or --no_subset was used (fact F6; look for any record: README, plots dir names like plots_patience5_subsetfrac0.1_dataset_bigvul, configs); class weights w=0.5/freq combined with focal gamma=2; baseline fairness: does CodeBERTOnly get the same sweep, same patience, same epochs, same focal loss, same threshold tuning, and is its head (CLS->200->2) comparable to the hybrid's (768->768->2 on fused vec)? pooling differences (CLS vs masked mean); patience 50 and max 200 epochs with train loss ~0 by epoch 50 (fact F3) - what does early stopping on val F1 at 0.5 threshold select?; single run / no seed control / no CI: how large is run-to-run F1 variance on BigVul in published work (search; cite), and is a 1.9-point gap distinguishable from noise; missing ablations (graph-only, random edges, no gate, frozen fusion) and what each would have told us; threshold search on val then test (fine) vs reporting; Table 1 has no accuracy though section 4.1 promises it. Classify each as methodology / thesis_code_mismatch / idea_limitation and whether it bears on code or idea.` },
  { key: 'dynamics', prompt: `${COMMON}
Finder key: dynamics. Scope: what the committed training curves say about why the hybrid underperforms. Use the Read tool to view these PNGs (each is small): ${REPO}/plots_Elevneth/{training_metrics,loss_curve,weights_curve,training_acc}_bs16_lr1e-04_br0p30.png, ${REPO}/plots_12/{training_metrics,loss_curve,weights_curve}_bs16_lr1e-04_br0p30.png, ${REPO}/plots_codebert_first/{training_metrics,loss_curve,weights_curve}_bs16_lr1e-04_br0p30.png, ${REPO}/plots_bigvul_attn1/training_metrics_bs16_lr1e-04_br0p30.png, ${REPO}/plots_bigvul_attn3/training_metrics_bs16_lr1e-04_br0p30.png, ${REPO}/plots_bigvul_attn5/training_metrics_bs16_lr1e-04_br0p30.png, ${REPO}/plots_bigvul_attn3/loss_curve_bs16_lr1e-04_br0p30.png, ${REPO}/plots_selfatention/{training_metrics,weights_curve,loss_curve}_bs16_lr1e-04_br0p30.png, ${REPO}/plots_attn_layer3_bs16_lr3e-05_br0p10/{training_metrics,loss_curve}_bs16_lr3e-05_br0p10.png, ${REPO}/plots_attn_layer2/training_metrics_bs8_lr5e-05_br0p20.png, ${REPO}/plots_attn_layer5/training_metrics_bs8_lr5e-05_br0p20.png, and a few of the val_p1_hist / val_conf_hist / confusion_matrix PNGs from plots_Elevneth at early (ep1, ep5), middle (ep50, ep100) and late (ep150, ep200) epochs (list the dir to get exact names). Also look at the sweep directories plots_patience5_subsetfrac0.1_dataset_bigvul and 'plots_patience=5_dataset=10%': list their training_metrics files, read about 6 spanning batch/lr/ratio, and summarise what the sweep found (which configs collapse to all-positive, which learn). Deliver evidence-backed findings: overfitting signature (train loss vs val loss), initial all-positive regime and how long it lasts, whether more fusion layers help (attn1/3/5), gate behaviour across runs (stable vs oscillating vs collapsed) and what that implies for the thesis section 5.2 interpretation and for the gate design, whether val F1 is still improving at epoch 200 or saturated, and anything in the histograms (probability mass near 0.5? bimodal?). Identify which run the thesis Figure 2 corresponds to and whether other runs with identical hyper-parameters contradict the thesis narrative. Each finding must say what it implies about code vs idea.` },
  { key: 'literature', prompt: `${COMMON}
Finder key: literature. Scope: is the idea (fuse a CPG graph encoder with CodeBERT via cross-attention for function-level vulnerability detection on BigVul) sound, and what does published evidence predict? Use WebSearch (mode "extended" for specific numbers) and WebFetch to check: (1) Vul-LMGNN / "Vul-LMGNNs: Fusing language models and online-distilled graph neural networks for code vulnerability detection" (2024) - datasets and F1; (2) DeepDFA + LineVul (Steenhoek et al., ICSE 2024 "Dataflow analysis-inspired deep learning for efficient vulnerability detection") - BigVul F1 of LineVul alone vs with DeepDFA, and the cross-project results; (3) LineVul (Fu and Tantithamthavorn, MSR 2022) BigVul F1 and the later critiques of that number (duplicates / PrimeVul by Ding et al. 2024 "Vulnerability detection with code language models: how far are we?" which reports CodeBERT-class models' F1 collapsing on PrimeVul); (4) Croft, Babar, Kholoosi ICSE 2023 "Data quality for software vulnerability datasets" - BigVul label accuracy / duplication / inconsistency percentages; (5) Steenhoek et al. ICSE 2023 "An empirical study of deep learning models for vulnerability detection" - run-to-run variance (F1 std across seeds) and token-vs-graph model comparison; (6) GALLa (ACL 2025) and EFVD (CNSSE 2025) cited by the thesis - what gains they report and on which datasets; (7) ReGVD / GraphCodeBERT / Devign BigVul numbers; (8) typical fine-tuned CodeBERT F1 on BigVul under random function-level splits in multiple papers (collect 3-5 numbers with their splits) to judge whether the thesis baseline (45.74) is in the normal range or indicates an under-powered setup (filter, subset, last-2-layers-only). Put the collected numbers in the summary field as a compact table with URLs and a verified/unverified flag per number. Then emit findings of category idea_limitation or methodology ONLY where the literature supports them, e.g. 'BigVul label noise caps achievable gains', 'published fusion gains on BigVul are small (X points) so a single-seed 1.9-point deficit is uninformative', 'structure helps mainly in cross-project / realistic splits', 'node features = CodeBERT CLS make the graph branch redundant with the text branch (compare with DeepDFA abstract dataflow features)'. Each finding must say what result a correct implementation of this idea would be expected to produce on BigVul.` },
  { key: 'provenance', prompt: `${COMMON}
Finder key: provenance. Scope: reproducibility and traceability of Table 1 / Figure 2: use git log / git show in ${REPO} (commits 6b1aa4c..a7122a4). Establish, with commit hashes and line evidence, which code state most plausibly produced the thesis numbers and what cannot be known. Check: HEAD cannot run the CodeBERT-only baseline (fact F1 NameError; confirm by reading run.py:460-495 and by a tiny Python simulation if you like); HEAD lacks the plotting code that made the figures; missing joern/graph-for-funcs.sc; hard-coded absolute /home/yudai paths; README refers to main.py but the file is run.py; requirements.txt vs actual imports (wandb, pyvis, seaborn, torch_geometric scatter deps) - would a fresh install even run; stale results.txt (26% positives) and what it is; W&B project name 'graduation-project-six' while --dataset defaults to 'six'; the 'primevul' branch where train/valid/test all point at the train jsonl (run.py around lines 1180-1183) - was it used?; wandb.init reinit per sweep config; whether the chosen config in thesis section 4.4 (bs16, lr1e-4, ratio 0.3) corresponds to any plots dir and whether the README sweep summary (which lacks ratio) comes from an older code version; configs.json epochs 100->200 change; the -path default and model file naming; whether the test-mode SAVE_PATH filename 'bert_rgcn.pth' matches the sweep's saved filename 'bert_rgcn_bs16_lr1e-04_br0p30.pth' (would '-mode test -path <dir>' load the right file?). Also list the dead/duplicated code paths that make the repo hard to trust (two test definitions, the import of utils.functions.cpg_dir - does that module exist? check; BalancedStreamDataset unused; legacy Devign Net/Conv). Emit reproducibility findings with severity reflecting whether they merely annoy or actually prevent anyone (including the student) from reproducing Table 1.` },
]

phase('Find')
log('Launching 7 finders')
const finderResults = (await parallel(FINDERS.map(f => () => agent(f.prompt, { label: `find:${f.key}`, phase: 'Find', schema: FINDINGS_SCHEMA })))).filter(Boolean)
const rawFindings = finderResults.flatMap(r => r.findings.map(x => ({ ...x, finder: r.finder })))
const finderSummaries = finderResults.map(r => ({ finder: r.finder, summary: r.summary }))
log(`Finders returned ${rawFindings.length} raw findings from ${finderResults.length}/7 finders`)

phase('Merge')
const merged = await agent(`You are merging audit findings about a graduation thesis and its code repository. Read ${BRIEF} for context. Below is a JSON array of raw findings from 7 independent finders. Merge duplicates (same underlying defect reported by several finders) into one finding that keeps the union of the evidence and the strongest well-supported severity; keep genuinely distinct findings separate even if they touch the same file; do not drop any distinct finding; do not invent new ones; keep ids traceable by listing merged source ids inside the claim text in square brackets. Keep each finding's bears_on and hurts consistent with its mechanism. Output every merged finding (do not cap the list). In the summary field, state how many raw findings became how many merged ones and list any you judged to be unsupported speculation (keep them out, but name them).

RAW FINDINGS JSON:
${JSON.stringify(rawFindings)}`, { label: 'merge', phase: 'Merge', schema: FINDINGS_SCHEMA })
const toVerify = merged ? merged.findings : rawFindings
log(`Merged to ${toVerify.length} findings; verifying each with 3 lenses`)

const LENSES = [
  { key: 'code', instr: `Lens: CODE TRACE. Re-read the cited code yourself (both the thesis-era snapshot and HEAD where relevant), follow every call path the claim depends on, and try hard to show the claim is false, overstated, or applies to a code version that did not produce the thesis numbers. A claim is refuted if the quoted mechanism does not actually happen in the code that plausibly produced Table 1 / Figure 2.` },
  { key: 'impact', instr: `Lens: IMPACT. Assume the claim is literally true. Argue whether it can materially change (a) the hybrid's F1, (b) the baseline's F1, (c) the validity of comparing them, or (d) the thesis's stated conclusions. Refute if the effect is negligible, applies equally to both models so it cannot explain the hybrid's deficit (say so explicitly - such findings may still be real but mislabeled), or if the finding's severity/category is wrong. Set severity_after accordingly.` },
  { key: 'empirical', instr: `Lens: EMPIRICAL. Test the claim by running something: write a small script under ${SCRATCH}/verify_<finding id>/ and run it with ${SCRATCH}/venv/bin/python (CPU torch, torch_geometric, transformers, pandas, numpy available; HuggingFace reachable; no dataset on disk, so fabricate minimal synthetic inputs or pickles that exercise the exact classes/functions from the repo by importing them from ${REPO} or ${SCRATCH}/thesis_era). Examples: instantiate the model and measure the thing claimed (param counts, gate behaviour under random init, R-GCN output norm after 6 layers, whether a NameError occurs, whether StreamGraphDataset yields identical order on two passes, whether validate() double-counts, what parse_to_functions produces from data/check_data/cpg_head_check.json). For literature / dataset-statistics claims, instead WebFetch the primary source and check the number. If the claim is not testable within about 15 minutes, say so, do not refute on that basis, and set confidence low. Paste the key program output into the evidence field.` },
]

function refutePrompt(f, lens) {
  return `You are an adversarial verifier. Read ${BRIEF} first. Then try to REFUTE the following audit finding about the thesis / repository. Default to refuted=true if, after real effort, the evidence does not hold up; default to refuted=false only when you have confirmed the mechanism. Be precise about which code version (thesis-era snapshot ${SCRATCH}/thesis_era vs HEAD ${REPO}) the claim holds for. If the claim is partly right, set refuted=false, write the corrected claim, and adjust severity_after. Your final answer is the structured verdict only.

${lens.instr}

FINDING:
${JSON.stringify(f, null, 2)}`
}

phase('Verify')
const verified = await parallel(toVerify.map(f => () =>
  parallel(LENSES.map(lens => () =>
    agent(refutePrompt(f, lens), { label: `verify:${f.id}:${lens.key}`, phase: 'Verify', schema: VERDICT_SCHEMA })
      .then(v => (v ? { lens: lens.key, ...v } : null))
  )).then(votes => ({ finding: f, votes: votes.filter(Boolean) }))
))
function survives(v) { return v.votes.length > 0 && v.votes.filter(x => x.refuted).length < 2 }
const surviving = verified.filter(survives)
const refutedList = verified.filter(v => !survives(v))
log(`Verification: ${surviving.length} survive, ${refutedList.length} refuted (of ${verified.length})`)

phase('Critic')
const critic = await agent(`You are the completeness critic for an audit whose question is: the thesis's hybrid model (R-GCN + CodeBERT via bidirectional cross-attention + gate) scored F1 43.85 vs CodeBERT baseline 45.74 on BigVul - is that because of defects in the code / experimental setup, or because the idea is unsound? Read ${BRIEF}, skim the thesis PDF at ${PDF} (Read tool, pages 1-7), and then read the verified findings and the refuted ones below. Tasks: (1) Give your own verdict on code_vs_idea with reasoning that weighs the surviving findings by mechanism and effect size, and explicitly separates 'the comparison in Table 1 is uninformative' from 'the hybrid is genuinely worse' from 'the idea cannot work'. (2) List contradictions between surviving findings or between a finding and the facts in the brief. (3) Rank the surviving finding ids by how much they matter to the student's question (most important first). (4) Propose MISSING findings: look for angles nobody covered (e.g. something in the thesis text itself that is internally inconsistent, a metric/number that cannot be right, a code path nobody traced, an experiment whose absence is decisive) - check the repo yourself before proposing; each missing finding needs file/line or page evidence like the others. Do not re-propose refuted ones unless you have new evidence (then say what is new). Final answer is the structured output only.

SURVIVING FINDINGS (with verifier votes):
${JSON.stringify(surviving.map(v => ({ finding: v.finding, votes: v.votes.map(x => ({ lens: x.lens, refuted: x.refuted, confidence: x.confidence, severity_after: x.severity_after, corrected_claim: x.corrected_claim, reasoning: x.reasoning })) })), null, 1)}

REFUTED FINDINGS (id, title, why):
${JSON.stringify(refutedList.map(v => ({ id: v.finding.id, title: v.finding.title, votes: v.votes.map(x => ({ lens: x.lens, refuted: x.refuted, reasoning: x.reasoning })) })), null, 1)}`, { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA })

let extraVerified = []
if (critic && critic.missing && critic.missing.length) {
  log(`Critic proposed ${critic.missing.length} additional findings; verifying`)
  extraVerified = await parallel(critic.missing.map(f => () =>
    parallel(LENSES.map(lens => () =>
      agent(refutePrompt(f, lens), { label: `verify2:${f.id}:${lens.key}`, phase: 'Critic', schema: VERDICT_SCHEMA })
        .then(v => (v ? { lens: lens.key, ...v } : null))
    )).then(votes => ({ finding: f, votes: votes.filter(Boolean) }))
  ))
}
const extraSurviving = extraVerified.filter(survives)
const extraRefuted = extraVerified.filter(v => !survives(v))
log(`Critic additions: ${extraSurviving.length} survive, ${extraRefuted.length} refuted`)

return {
  finderSummaries,
  mergeSummary: merged ? merged.summary : null,
  counts: { raw: rawFindings.length, merged: toVerify.length, surviving: surviving.length, refuted: refutedList.length, extraSurviving: extraSurviving.length, extraRefuted: extraRefuted.length },
  surviving: surviving.concat(extraSurviving),
  refuted: refutedList.concat(extraRefuted).map(v => ({ id: v.finding.id, title: v.finding.title, claim: v.finding.claim, votes: v.votes.map(x => ({ lens: x.lens, refuted: x.refuted, confidence: x.confidence, reasoning: x.reasoning, evidence: x.evidence })) })),
  critic: critic ? { verdict: critic.verdict, code_vs_idea: critic.code_vs_idea, reasoning: critic.reasoning, contradictions: critic.contradictions, ranked_finding_ids: critic.ranked_finding_ids } : null,
}