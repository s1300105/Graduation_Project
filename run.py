import argparse
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import shutil
import gc
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ★ PyG の DataLoader を使う
from torch_geometric.loader import DataLoader as GeoDataLoader

import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_dir as cpg  # 参照されているので残す
from utils.functions.cpg import parse_to_functions
from utils.process.embeddings import nodes_to_input
from utils.data.datamanager import loads, train_val_test_split  # 互換のため残す（未使用）
from utils.validate.save_utils import save_split_from_loaders, cache_loader_items, save_loader_sample_json
from utils.validate.analyze_utils import peek_loader, summarize_generic_loader, summarize_graph_loader, describe_pyg_graph

from models.LMGNN import BertRGCN
from models.layers import Conv
from test import test
import os, gc, random
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Batch
from utils.data.balanced_iter import BalancedBatcher
from torch_geometric.loader import DataLoader as GeoDataLoader
from collections import deque
import numpy as np
from transformers import get_linear_schedule_with_warmup

PATHS = configs.Paths()
FILES = configs.Files()

# 公式配布の分割データ
train_path = "/home/yudai/Project/research/Vul_Detection/data/raw/new_six_by_projects/train.jsonl"
valid_path = "/home/yudai/Project/research/Vul_Detection/data/raw/new_six_by_projects/valid.jsonl"
test_path  = "/home/yudai/Project/research/Vul_Detection/data/raw/new_six_by_projects/test.jsonl"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




class BalancedStreamDataset(IterableDataset):
    def __init__(self, files, *, batch_size: int, buffer_size: int = 256, shuffle: bool = True):
        super().__init__()
        self.files = files
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.batch_size = int(batch_size)
        assert self.batch_size >= 2 and (self.batch_size % 2 == 0), "batch_size は偶数を推奨"

    def _inner_dataset(self):
        return StreamGraphDataset(self.files, buffer_size=self.buffer_size, shuffle=self.shuffle)

    def __iter__(self):
        inner = iter(self._inner_dataset())
        need0 = self.batch_size // 2
        need1 = self.batch_size - need0
        buf0, buf1 = deque(maxlen=self.buffer_size), deque(maxlen=self.buffer_size)

        while True:
            # まず必要数までバッファを補充
            try:
                while len(buf0) < need0 or len(buf1) < need1:
                    g = next(inner)  # ← ここで尽きることがある
                    y = int(g.y.item() if torch.is_tensor(g.y) else int(g.y))
                    (buf1 if y == 1 else buf0).append(g)
            except StopIteration:
                # ここから「後始末」：作れるだけバランス塊を吐く
                while len(buf0) >= need0 and len(buf1) >= need1:
                    batch = []
                    for _ in range(need0):
                        batch.append(buf0.popleft())
                    for _ in range(need1):
                        batch.append(buf1.popleft())
                    random.shuffle(batch)
                    for gg in batch:
                        yield gg

                # 端数（どちらかだけ余った分）も落としたくない場合は吐き切る
                # 端数を評価に載せたくないなら、ここをコメントアウトして
                # 外側 DataLoader の drop_last=True にする
                tail = []
                if buf0:
                    tail.extend(buf0)
                    buf0.clear()
                if buf1:
                    tail.extend(buf1)
                    buf1.clear()
                if tail:
                    random.shuffle(tail)
                    for gg in tail:
                        yield gg

                # 正常終了（StopIteration を外へ投げない）
                return

            # 通常系：1塊分を作って吐く
            batch = []
            for _ in range(need0):
                batch.append(buf0.popleft())
            for _ in range(need1):
                batch.append(buf1.popleft())
            random.shuffle(batch)
            for gg in batch:
                yield gg



# ------------------------------------------------------------
# 公式 split 前提の select（qemu 固定などの旧ロジックは無効化）
# ------------------------------------------------------------
def select(dataset: pd.DataFrame) -> pd.DataFrame:
    # 関数長フィルタのみ（必要に応じて変更）
    len_filter = dataset.func.str.len() < 1200
    result = dataset.loc[len_filter]
    # 開発中に重すぎる時の上限（必要なければコメントアウト）
    #result = result.head(10000)
    return result

def shuffle_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """DataFrame をシャッフルし、index をリセットして返す"""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)




# ------------------------------------------------------------
# 公式 split ごとの CPG 生成（.bin -> graph JSON -> join -> pkl）
# ------------------------------------------------------------
def CPG_generator():
    context = configs.Create()

    if Path(PATHS.cpg).exists():  
        shutil.rmtree(PATHS.cpg)  
    Path(PATHS.cpg).mkdir(parents=True, exist_ok=True)  

    def _read_jsonl(fp: str) -> pd.DataFrame:
        try:
            return pd.read_json(fp, lines=True)
        except ValueError:
            # もし配列形式なら
            return pd.read_json(fp, lines=False)



    train_raw = shuffle_df(_read_jsonl(train_path), seed=42)
    valid_raw = shuffle_df(_read_jsonl(valid_path), seed=43)
    test_raw  = shuffle_df(_read_jsonl(test_path),  seed=44)


    train_filtered = data.apply_filter(train_raw, select)
    valid_filtered = data.apply_filter(valid_raw, select)
    test_filtered  = data.apply_filter(test_raw,  select)

    train_filtered = data.clean(train_filtered)
    valid_filtered = data.clean(valid_filtered)
    test_filtered  = data.clean(test_filtered)

    # 公式 split なので commit_id / project は落としておく（必要なら残してOK）
    data.drop(train_filtered, ["commit_id", "project"])
    data.drop(valid_filtered, ["commit_id", "project"])
    data.drop(test_filtered,  ["commit_id", "project"])

    def _process_split(split_name: str, df_filtered: pd.DataFrame):
        """
        split_name : "train" / "valid" / "test"
        df_filtered: select/clean 済み DataFrame（index が関数IDとして使われる）
        """
        # スライス（スライスID s と、その中身 slice_df）
        slices = data.slice_frame(df_filtered, context.slice_size)
        slices = [(s, slice_df.apply(lambda x: x)) for s, slice_df in slices]

        cpg_files = []
        for s, slice_df in slices:
            # ① Joern 入力コードをファイル群に展開
            data.to_files(slice_df, PATHS.joern)
            # ② CPG 生成（*.bin）
            out_stem = f"{split_name}_{s}_{FILES.cpg}"  # 例: train_0_cpg
            cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, out_stem)
            cpg_files.append(cpg_file)
            print(f"[{split_name}] Slice {s} -> CPG: {cpg_file}")
            # ③ 一時コード群を掃除
            shutil.rmtree(PATHS.joern, ignore_errors=True)

        if not cpg_files:
            print(f"[{split_name}] No CPG files produced.")
            return

        # ④ CPG(.bin) -> JSON へ（graph-for-funcs.sc 実行）
        json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
        if not json_files:
            print(f"[{split_name}] No JSON graphs produced.")
            return

        # ⑤ JSON を読み、Index で join して pkl 保存
        for (s, slice_df), json_file in zip(slices, json_files):
            graphs = process.json_process(PATHS.cpg, json_file)
            if not graphs:
                print(f"[{split_name}] Slice {s} JSON empty. Skipped.")
                continue

            df_graphs = pd.DataFrame(graphs, columns=["Index", "cpg"])
            df_graphs["Index"] = df_graphs["Index"].astype(int)

            # 左側: slice_df に "Index" が無いケースも想定して安全に作る
            left_cols = ["Index", "func"] if "Index" in slice_df.columns else slice_df.columns.tolist()
            left  = data.create_with_index(slice_df, left_cols)
            right = data.create_with_index(df_graphs, ["Index", "cpg"])
            dataset = data.inner_join_by_index(left, right)

            out_name = f"{split_name}_{s}_{FILES.cpg}.pkl"
            print(f"[{split_name}] Writing: {out_name} (rows={len(dataset)})")
            data.write(dataset, PATHS.cpg, out_name)
            del dataset
            gc.collect()

    # ------ 各 split を順番に処理 ------
    _process_split("train", train_filtered)
    _process_split("valid", valid_filtered)
    _process_split("test",  test_filtered)


# ------------------------------------------------------------
# CPG -> PyG Data 入力（CodeBERT埋め込みを含む）
# ------------------------------------------------------------
def Embed_generator(resume: bool = False):
    context = configs.Embed()

    # --- ディレクトリの扱い ---
    if not resume:
        # フル再生成モード → 既存の input/tokens を全部消す
        if Path(PATHS.input).exists():
            shutil.rmtree(PATHS.input)
        Path(PATHS.input).mkdir(parents=True, exist_ok=True)

        if Path(PATHS.tokens).exists():
            shutil.rmtree(PATHS.tokens)
        Path(PATHS.tokens).mkdir(parents=True, exist_ok=True)
    else:
        # 再開モード → 既存は消さずに、そのまま使う
        Path(PATHS.input).mkdir(parents=True, exist_ok=True)
        Path(PATHS.tokens).mkdir(parents=True, exist_ok=True)

    # すでに作成ずみの *_input.pkl の stem を集める
    existing_input_stems = set()
    for f in data.get_directory_files(PATHS.input):  # 例: ["train_0_input.pkl", ...]
        if f.endswith(f"_{FILES.input}"):
            # 例: "train_0_input.pkl" → "train_0_input" → "train_0_cpg" と合わせたいなら
            stem = f[: -len(f"_" + FILES.input)]  # "train_0"
            existing_input_stems.add(stem)

    dataset_files = data.get_directory_files(PATHS.cpg)  # ["train_0_cpg.pkl", "train_1_cpg.pkl", ...]

    for pkl_file in dataset_files:
        # 例: "train_0_cpg.pkl" → "train_0_cpg" → "train_0"
        file_stem_full = pkl_file.split(".")[0]       # "train_0_cpg"
        base_stem = file_stem_full.rsplit("_", 1)[0]  # "train_0"

        # 再開モードのとき、既に対応する *_input.pkl があればスキップ
        if resume and base_stem in existing_input_stems:
            print(f"[RESUME] skip already processed: {base_stem}")
            continue

        print(f"[INFO] processing CPG: {pkl_file}")
        cpg_dataset = data.load(PATHS.cpg, pkl_file)

        # --- トークンも再開したければ同様にスキップ処理を入れてOK ---
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_stem_full}_{FILES.tokens}")

        rows = []
        for _, row in cpg_dataset.iterrows():
            func_dicts = parse_to_functions(row.cpg, max_nodes=context.nodes_dim)
            for nodes in func_dicts:
                g = process.nodes_to_input(nodes, row.target, context.nodes_dim, mode="codebert")
                g.func = row.func

                try:
                    if "commit_id" in row:
                        setattr(g, "commit_id", row["commit_id"])
                except Exception:
                    pass
                try:
                    _idx = int(row["Index"]) if "Index" in row and pd.notna(row["Index"]) else int(row.get("idx", -1))
                    setattr(g, "idx", _idx)
                except Exception:
                    setattr(g, "idx", -1)

                out_row = {
                    "input": g,
                    "target": row.target,
                    "func": row.func if "func" in row else None,
                }
                if "commit_id" in row:
                    out_row["commit_id"] = row["commit_id"]
                if "Index" in row:
                    try:
                        out_row["idx"] = int(row["Index"])
                    except Exception:
                        pass
                elif "idx" in row:
                    out_row["idx"] = row["idx"]

                rows.append(out_row)

        base_cols = ["input", "target", "func"]
        extra_cols = []
        if rows:
            sample_keys = set().union(*[r.keys() for r in rows])
            for k in ("commit_id", "idx"):
                if k in sample_keys:
                    extra_cols.append(k)

        out_df = pd.DataFrame(rows, columns=base_cols + extra_cols)
        out_name = f"{base_stem}_{FILES.input}"   # 例: "train_0_input.pkl"
        print(f"[INFO] saving INPUT rows={len(out_df)}  file={out_name}")
        if len(out_df) == 0:
            print("[WARN] No functions parsed from this CPG chunk. Check parser/filter settings.")

        data.write(out_df, PATHS.input, out_name)



def train(model, train_loader, optimizer, epoch, criterion, scheduler=None):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        try:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(batch)          # [B, 2]
            batch.y = batch.y.view(-1).long()
            loss = criterion(y_pred, batch.y)
            loss.backward()

            if (batch_idx+1) % 200 == 0:
                _diag_logits("train", y_pred, batch.y)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if (batch_idx + 1) % 100 == 0:
                num_graphs = getattr(batch, "num_graphs", 1)
                print(f'Train Epoch: {epoch} step {batch_idx+1} (+{num_graphs} graphs)\tLoss: {loss.item():.6f}')
        except RuntimeError as e:
            # まれな CUDA OOM をスキップして進行
            if "out of memory" in str(e).lower():
                print(f"[WARN] CUDA OOM at step {batch_idx+1}. Skipping this batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise


# ==== 修正済み validate ====
def validate(model, val_loader):
    if val_loader is None:
        print("[WARN] validate() called with None loader. Skipped.")
        return 0.0, 0.0, 0.0, 0.0

    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true, y_pred_labels = [], []

    # ★ ロジット診断の状態を初期化
    diag_state = _diag_logits_init()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)
            logits = model(batch)                 # [B, 2]
            batch.y = batch.y.view(-1).long()
            loss = F.cross_entropy(logits, batch.y)

            total_loss += loss.item()
            n_batches += 1

            # 予測ラベル収集
            pred = logits.argmax(dim=-1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred_labels.extend(pred.cpu().tolist())

            # ★ ロジット診断を逐次更新
            _diag_logits_update(diag_state, logits, batch.y)

    # IterableDataLoader 対応: バッチ数で割る
    avg_loss = total_loss / max(1, n_batches)

    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)

    # 混同行列を保存
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Val)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_val.png')
    plt.close()

    # ★ ロジット診断を出力（平均確率/信頼度・クラス偏りなど）
    _diag_logits_report("val", diag_state)

    print('Val: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        avg_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100
    ))
    return accuracy, precision, recall, f1
# ==== ここまで ====



def test(model, test_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true, y_pred_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            logits = model(batch)
            batch.y = batch.y.view(-1).long()
            total_loss += F.cross_entropy(logits, batch.y).item()
            n_batches += 1

            pred = logits.argmax(dim=-1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred_labels.extend(pred.cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)

    print('Test: Average loss: {:.4f}, Acc: {:.2f}%, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%'.format(
        avg_loss, accuracy*100, precision*100, recall*100, f1*100
    ))
    return accuracy, precision, recall, f1


def _collect_probs_and_labels(model, loader):
    """loader から class=1 の確率とラベルを全部集めるヘルパ"""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)[:, 1]  # class=1 の確率
            labels = batch.y.view(-1).long()

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    return np.array(all_probs), np.array(all_labels), avg_loss


def search_best_threshold(model, val_loader, num_steps: int = 19):
    """Validation 上で F1 が最大になるしきい値を探索"""
    if val_loader is None:
        print("[ThreshSearch] val_loader is None; use default threshold=0.5")
        return 0.5, None

    probs, labels, _ = _collect_probs_and_labels(model, val_loader)

    best_f1 = -1.0
    best_t = 0.5
    best_stats = None

    for t in np.linspace(0.05, 0.95, num_steps):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_stats = (acc, prec, rec, f1)

    if best_stats is not None:
        acc, prec, rec, f1 = best_stats
        print(
            f"[ThreshSearch] best_t={best_t:.2f} | "
            f"Val Acc={acc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}"
        )
    else:
        print("[ThreshSearch] failed to search threshold, fallback to 0.5")

    return best_t, best_stats


def evaluate_with_threshold(model, loader, threshold: float):
    """指定しきい値で評価する"""
    probs, labels, avg_loss = _collect_probs_and_labels(model, loader)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return avg_loss, acc, prec, rec, f1






class StreamGraphDataset(IterableDataset):
    """
    <split>_*.pkl を逐次読み込み → 小さめバッファに貯めて（疑似）シャッフル → yield
    * buffer_size は batch_size×8〜16 が目安（OOM 回避のため小さめ）
    """
    def __init__(self, files, *, buffer_size=128, shuffle=True):
        super().__init__()
        self.files = list(sorted(files))
        self.buffer_size = int(buffer_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        rng = random.Random(42 + (os.getpid() if hasattr(os, "getpid") else 0))
        buffer = []
        for fn in self.files:
            df = pd.read_pickle(fn)  # （.pkl 形式想定）
            try:
                # itertuples は iterrows より低オーバーヘッド
                for row in df.itertuples(index=False):
                    # row は列名を属性に持つ namedtuple
                    g = getattr(row, "input", None)
                    if g is None:
                        continue
                    try:
                        y = int(getattr(row, "target", 0))
                    except Exception:
                        y = 0
                    g.y = torch.tensor([y], dtype=torch.long)

                    buffer.append(g)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            rng.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer.clear()
            finally:
                # 速やかな解放
                del df
                gc.collect()

        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            for item in buffer:
                yield item
            buffer.clear()

def _diag_logits(name, logits, y):
    with torch.no_grad():
        probs = logits.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        frac_pos = float((pred == 1).float().mean().item())  # 1予測の割合
        avg = probs.mean(dim=0).tolist()
        print(f"[Diag:{name}] p_mean={avg} frac_pred1={frac_pos:.3f} "
              f"y_mean={float((y.float().mean().item())):.3f}")
        
# ==== logits 診断ヘルパ ====
def _diag_logits_init():
    # 逐次集計用の状態
    return {
        "sum_p1": 0.0,        # 予測確率(クラス1)の和
        "sum_conf": 0.0,      # 予測信頼度(max softmax)の和
        "sum_pred1": 0,       # 予測が1の個数
        "sum_label1": 0,      # 正解が1の個数
        "n": 0,               # サンプル総数
        "p1_vals": [],        # （任意）確率ヒスト用に一部だけ溜める
        "conf_vals": [],      # （任意）信頼度ヒスト用
        "cap": 20000,         # 可視化のために最大2万件だけバッファ
    }

def _diag_logits_update(state, logits: torch.Tensor, labels: torch.Tensor):
    # logits: [B, 2]（CE想定）/ labels: [B]
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)           # [B,2]
        p1 = probs[:, 1]                                 # [B]
        conf, pred = probs.max(dim=-1)                   # [B], [B]

        b = labels.numel()
        state["sum_p1"]    += float(p1.sum().item())
        state["sum_conf"]  += float(conf.sum().item())
        state["sum_pred1"] += int((pred == 1).sum().item())
        state["sum_label1"]+= int(labels.sum().item())
        state["n"]         += int(b)

        # ヒスト可視化のために上限付きで蓄積（CPU側へ）
        if state["cap"] > 0:
            take = min(b, state["cap"])
            state["p1_vals"].extend(p1[:take].detach().cpu().tolist())
            state["conf_vals"].extend(conf[:take].detach().cpu().tolist())
            state["cap"] -= take

def _diag_logits_report(stage: str, state):
    if state["n"] == 0:
        print(f"[diag:{stage}] no samples")
        return
    mean_p1   = state["sum_p1"]   / state["n"]
    mean_conf = state["sum_conf"] / state["n"]
    frac_pred1  = state["sum_pred1"]  / state["n"]
    frac_label1 = state["sum_label1"] / state["n"]

    print(
        f"[diag:{stage}] n={state['n']}, "
        f"mean p(class=1)={mean_p1:.4f}, "
        f"mean confidence={mean_conf:.4f}, "
        f"pred#1 rate={frac_pred1*100:.2f}%, "
        f"label#1 rate={frac_label1*100:.2f}%"
    )

    # （任意）簡易ヒストグラムをPNG保存
    try:
        import matplotlib.pyplot as plt
        if state["p1_vals"]:
            plt.figure(figsize=(4,3))
            plt.hist(state["p1_vals"], bins=20)
            plt.title(f'P(class=1) Histogram [{stage}]')
            plt.tight_layout()
            plt.savefig(f'val_p1_hist_{stage}.png')
            plt.close()
        if state["conf_vals"]:
            plt.figure(figsize=(4,3))
            plt.hist(state["conf_vals"], bins=20)
            plt.title(f'Confidence Histogram [{stage}]')
            plt.tight_layout()
            plt.savefig(f'val_conf_hist_{stage}.png')
            plt.close()
    except Exception:
        pass
# ==== ここまでヘルパ ====



def pyg_collate(batch_list):
    return Batch.from_data_list(batch_list)


def make_stream_loader_for_split(split: str, *, batch_size: int, buffer_size: int, shuffle: bool):
    files = _split_file_paths(split)
    if not files:
        return None
    ds = StreamGraphDataset(files, buffer_size=buffer_size, shuffle=shuffle)
    # まずはワーカ0（プロセス分の常駐メモリを避ける）
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=pyg_collate,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def _resolve_save_path(user_path: str | None, default_dir: str = "./saved_models", filename: str = "bert_rgcn.pth") -> Path:
    if not user_path or user_path.strip() == "":
        save_dir = Path(default_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / filename
    p = Path(user_path)
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
        return p / filename
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    

def _split_file_paths(split: str):
    all_files = data.get_directory_files(PATHS.input)  # list[str]
    return [str(Path(PATHS.input) / f)
            for f in all_files
            if f.startswith(f"{split}_") and f.endswith(f"_{FILES.input}")]

def _count_targets_in_files(files, require_input: bool = False):
    """
    files に含まれる train_*.pkl を走査して、
    ・ラベル0/1の個数（input がある行だけを対象にできる）
    を数えるヘルパ。
    """
    cnt0 = cnt1 = 0
    for fn in files:
        try:
            df = pd.read_pickle(fn)
            if "target" not in df.columns:
                continue

            # ★ 学習に実際に使うのは input がある行だけなので、その行だけに絞る
            if require_input and "input" in df.columns:
                mask = df["input"].notna()
                df = df.loc[mask]
                if df.empty:
                    continue

            vals = df["target"].astype("int64").clip(0, 1)
            c1 = int(vals.sum())
            c0 = int(len(vals) - c1)
            cnt0 += c0
            cnt1 += c1

            del df
            gc.collect()
        except Exception:
            continue
    return cnt0, cnt1


# ------------------------------------------------------------
# 公式 split 固定のローダ作成（*_input.pkl を読み込む）
# ------------------------------------------------------------
def _load_split_inputs(split: str):
    assert split in {"train", "valid", "test"}
    all_files = data.get_directory_files(PATHS.input)  # ディレクトリ内ファイル名 list[str]
    pick = [f for f in all_files if f.startswith(f"{split}_") and f.endswith(f"_{FILES.input}")]
    if not pick:
        print(f"[WARN] No input files for split={split}. Did you run --cpg and then --embed?")
        return []
    import pandas as pd
    out = []
    for fn in sorted(pick):
        df = data.load(PATHS.input, fn)  # DataFrame: ["input","target","func",("commit_id","idx" ...)]
        for _, row in df.iterrows():
            g = row.get("input", None)
            if g is None:
                continue
            # y（ラベル）を確実に付与
            try:
                y = int(row.get("target", 0))
            except Exception:
                y = 0
            try:
                g.y = torch.tensor([y], dtype=torch.long)
            except Exception:
                continue
            # メタ
            if "idx" in row and pd.notna(row["idx"]):
                try: g.idx = int(row["idx"])
                except Exception: pass
            if "commit_id" in row and pd.notna(row["commit_id"]):
                try: g.commit_id = str(row["commit_id"])
                except Exception: pass
            if "func" in row and pd.notna(row["func"]):
                try: g.func = row["func"]
                except Exception: pass
            out.append(g)
    print(f"[SplitLoad] {split}: {len(out)} graphs")
    return out

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpg', '--cpg', action='store_true', help='CPG generation task')
    parser.add_argument('-embed', '--embed', action='store_true', help='Embedding generation task')
    parser.add_argument('-mode', '--mode', default="train", help='train / test')
    parser.add_argument('-path', '--path', default=None, help='model path or dir')
    args = parser.parse_args()

    if args.cpg:
        CPG_generator()
    if args.embed:
        Embed_generator(resume=False)

    # === 公式 split 固定で入力をロード ===
    context = configs.Process()

    # --- train: ストリーミング用のファイルリスト ---
    train_files = _split_file_paths("train")
    if args.mode == "train" and not train_files:
        raise RuntimeError("No training data found. Run with --cpg そして --embed を済ませてください。")

    # ★ 学習に実際に使う「有効サンプル数」を数える
    #   - input が存在する行だけを対象にして
    #   - BalancedStreamDataset で 0/1=1:1 にするので、1エポック ≒ 2 * min(cnt0, cnt1)
    cnt0, cnt1 = _count_targets_in_files(train_files, require_input=True)
    total_samples_raw = max(1, cnt0 + cnt1)

    # ★ クラス重みを計算（頻度の逆数ベース）
    #   - 片方が多いほど、そのクラスの重みを小さくする
    freq0 = cnt0 / total_samples_raw
    freq1 = cnt1 / total_samples_raw
    w0 = 0.5 / freq0    # 0.5 は「両クラスの目標比率」
    w1 = 0.5 / freq1

    print(f"[ClassCount] num0={cnt0}, num1={cnt1}, "
        f"raw_total={total_samples_raw}, w0={w0:.4f}, w1={w1:.4f}")

    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    print("Using device:", device)
    print("Class weights:", class_weights)

    # === モデル設定読み込み ===========================================
    BertModelCfg = configs.BertGGNN()
    gated_graph_conv_args = BertModelCfg.model["gated_graph_conv_args"]
    conv_args = BertModelCfg.model["conv_args"]
    emb_size = BertModelCfg.model["emb_size"]

    # ------------------------------------------------------------
    # TRAIN モード：バッチサイズ × 学習率 の自動スイープ + Warmup + EarlyStopping
    # ------------------------------------------------------------
    if args.mode == "train":
        if not train_files:
            raise RuntimeError("train split が空です。")

        # 探索するバッチサイズと学習率の候補
        batch_candidates = [8, 16, 32]
        lr_candidates = [1e-4, 5e-5, 3e-5, 1e-5]

        patience = 5
        NUM_EPOCHS = context.epochs

        all_results = []  # (batch, lr, best_f1, best_epoch) を記録

        for batch_size in batch_candidates:
            print("=" * 80)
            print(f"[SWEEP] Start batch_size={batch_size}")
            print("=" * 80)

            # --- train_loader（BalancedStreamDataset を使用） ---
            """
            balanced_ds = BalancedStreamDataset(
                train_files,
                batch_size=batch_size,
                buffer_size=max(128, batch_size * 8),
                shuffle=True,
            )
            train_loader = GeoDataLoader(
                balanced_ds,
                batch_size=batch_size,
                collate_fn=pyg_collate,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
                shuffle=False,  # BalancedStreamDataset 側で順序制御する
            )
            """

            # --- train_loader（全データ＋クラス重みで不均衡対策） ---
            train_loader = make_stream_loader_for_split(
                "train",
                batch_size=batch_size,
                buffer_size=max(128, batch_size * 8),
                shuffle=True,   # ← train は真で OK
            )


            # --- val_loader ---
            val_loader = make_stream_loader_for_split(
                "valid",
                batch_size=batch_size,
                buffer_size=max(64, batch_size * 4),
                shuffle=False
            )

            if val_loader is None:
                print("[WARN] valid split が空です。検証なしで学習を続行します。")

            # 学習率スイープ
            for lr in lr_candidates:
                print("-" * 60)
                print(f"[SWEEP] batch_size={batch_size}, lr={lr}")
                print("-" * 60)

                # モデルを新規に構築
                model = BertRGCN(gated_graph_conv_args, conv_args, emb_size, device, Conv=Conv).to(device)

                 # Optimizer
                wd = 5e-4
                print(f"[OPTIM] AdamW lr={lr}, weight_decay={wd}")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                )

                # Scheduler（Warmup + Linear Decay）
                # ★ IterableDataset なので len(train_loader) ではなく、
                #   BalancedStreamDataset の実効サンプル数を使って steps_per_epoch を近似
                steps_per_epoch = math.ceil(total_samples_raw / batch_size)
                total_steps = steps_per_epoch * NUM_EPOCHS
                warmup_steps = int(0.1 * total_steps)

                if total_steps > 0:
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                    )
                    print(f"[SCHED] total_steps={total_steps}, warmup_steps={warmup_steps}, "
                        f"steps_per_epoch≈{steps_per_epoch}")
                else:
                    scheduler = None
                    print("[WARN] total_steps が 0 のため Scheduler 無効")

                # EarlyStopping 用
                best_f1 = 0.0
                best_epoch = 0
                wait = 0

                # ログ用 history（この組み合わせ専用）
                history = {"epoch": [], "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": []}

                # 保存ファイル名（バッチとLRを含める）
                lr_str = f"{lr:.0e}"  # 例: 1e-04
                SAVE_PATH = _resolve_save_path(
                    args.path,
                    filename=f"bert_rgcn_bs{batch_size}_lr{lr_str}.pth"
                )

                for epoch in range(1, NUM_EPOCHS + 1):
                    # --- 学習 ---
                    train(model, train_loader, optimizer, epoch, criterion, scheduler=scheduler)

                    if val_loader is not None:
                        # ① まずは従来どおり「argmax（しきい値0.5）」で評価
                        acc, precision, recall, f1 = validate(model, val_loader)

                        history["epoch"].append(epoch)
                        history["val_acc"].append(acc)
                        history["val_prec"].append(precision)
                        history["val_rec"].append(recall)
                        history["val_f1"].append(f1)  # これは F1@0.5

                        # ② Validation 上で「F1が最大になるしきい値」を探索
                        best_t, best_stats = search_best_threshold(model, val_loader, num_steps=19)

                        if best_stats is not None:
                            acc_t, prec_t, rec_t, f1_t = best_stats
                            f1_for_es = f1_t        # EarlyStopping 用
                            used_threshold = best_t # ログ用
                        else:
                            # 念のため、探索に失敗したときは従来の F1 を使う
                            f1_for_es = f1
                            used_threshold = 0.5


                        # ④ F1 改善チェック（EarlyStopping は「ベストしきい値の F1」で判定）
                        if f1_for_es > best_f1 + 1e-4:
                            best_f1 = f1_for_es
                            best_epoch = epoch
                            wait = 0
                            torch.save(model.state_dict(), str(SAVE_PATH))
                            saved_flag = " [SAVED]"
                        else:
                            wait += 1
                            saved_flag = ""

                        # ⑤ ログには「F1@0.5」と「F1@best_t」の両方を出す
                        print(
                            f"Epoch {epoch}/{NUM_EPOCHS} | "
                            f"Acc@0.5: {acc:.4f}, P@0.5: {precision:.4f}, R@0.5: {recall:.4f}, F1@0.5: {f1:.4f}, "
                            f"F1@best_t({used_threshold:.2f}): {f1_for_es:.4f}, "
                            f"best F1: {best_f1:.4f} (epoch {best_epoch}){saved_flag} "
                            f"wait={wait}/{patience}"
                        )

                        if wait >= patience:
                            print(f"[EarlyStopping] no F1 improvement for {patience} epochs. Stop training.")
                            break

                    else:
                        # val_loader が無い場合は毎エポック上書きで保存
                        torch.save(model.state_dict(), str(SAVE_PATH))


                # この (batch_size, lr) の結果を記録
                all_results.append({
                    "batch_size": batch_size,
                    "lr": lr,
                    "best_f1": float(best_f1),
                    "best_epoch": int(best_epoch),
                    "model_path": str(SAVE_PATH),
                })

                # 簡単な学習曲線を保存 (組み合わせごとに別ファイル)
                if history["epoch"]:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(6,4))
                    plt.plot(history["epoch"], history["val_f1"], label="Val F1")
                    plt.plot(history["epoch"], history["val_rec"], label="Val Recall")
                    plt.plot(history["epoch"], history["val_prec"], label="Val Precision")
                    plt.xlabel("Epoch")
                    plt.ylabel("Score")
                    plt.ylim(0.0, 1.0)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"training_metrics_bs{batch_size}_lr{lr_str}.png")
                    plt.close()

                    plt.figure(figsize=(6,4))
                    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.ylim(0.0, 1.0)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"training_acc_bs{batch_size}_lr{lr_str}.png")
                    plt.close()

        # 全体の結果を表示
        print("\n===== SWEEP SUMMARY (sorted by best_f1) =====")
        for r in sorted(all_results, key=lambda x: x["best_f1"], reverse=True):
            print(
                f"batch={r['batch_size']}, lr={r['lr']}, "
                f"best_f1={r['best_f1']:.4f} (epoch {r['best_epoch']}), "
                f"model={r['model_path']}"
            )

    # ------------------------------------------------------------
    # TEST モード：指定モデルでしきい値サーチ + 評価
    # ------------------------------------------------------------
    if args.mode == "test":
        # test_loader / val_loader を作成（batch_size は context.batch_size を利用）
        test_loader = make_stream_loader_for_split(
            "test",
            batch_size=context.batch_size,
            buffer_size=max(64, context.batch_size * 4),
            shuffle=False
        )
        val_loader = make_stream_loader_for_split(
            "valid",
            batch_size=context.batch_size,
            buffer_size=max(64, context.batch_size * 4),
            shuffle=False
        )
        if test_loader is None:
            raise RuntimeError("test split が空です。")

        # SAVE_PATH は単一モデルを指す（従来通り）
        SAVE_PATH = _resolve_save_path(args.path, filename="bert_rgcn.pth")

        model_test = BertRGCN(gated_graph_conv_args, conv_args, emb_size, device, Conv=Conv).to(device)
        model_test.load_state_dict(torch.load(str(SAVE_PATH), map_location=device))

        # ① 固定しきい値 0.5
        loss0, acc0, p0, r0, f10 = evaluate_with_threshold(model_test, test_loader, threshold=0.5)
        print('Test (thr=0.50): Average loss: {:.4f}, Acc: {:.2f}%, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%'.format(
            loss0, acc0*100, p0*100, r0*100, f10*100
        ))

        # ② Validation でベストしきい値を探索
        best_t, _ = search_best_threshold(model_test, val_loader)

        # ③ そのしきい値で Test を評価
        loss, acc, p, r, f1 = evaluate_with_threshold(model_test, test_loader, threshold=best_t)
        print('Test (best thr={:.2f} from Val): Average loss: {:.4f}, Acc: {:.2f}%, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%'.format(
            best_t, loss, acc*100, p*100, r*100, f1*100
        ))
