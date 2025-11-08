import argparse
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import shutil
import gc
import os

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ★ 追加：PyG の DataLoader を使う
from torch_geometric.loader import DataLoader as GeoDataLoader

import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_dir as cpg  # 参照されているので残す
from utils.functions.cpg import parse_to_functions
from utils.process.embeddings import nodes_to_input
from utils.data.datamanager import loads, train_val_test_split
from utils.validate.save_utils import save_split_from_loaders, cache_loader_items, save_loader_sample_json
from utils.validate.analyze_utils import peek_loader, summarize_generic_loader, summarize_graph_loader, describe_pyg_graph

from models.LMGNN import BertGGCN
from test import test


PATHS = configs.Paths()
FILES = configs.Files()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def select(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.loc[dataset['project'] == "qemu"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    result = result.head(1000)
    return result


def CPG_generator():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)  # DataFrame

    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)

    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if not graphs:
            print(f"Dataset chunk {s} not processed.")
            continue

        df_graphs = pd.DataFrame(graphs, columns=["Index", "cpg"])
        df_graphs["Index"] = df_graphs["Index"].astype(int)

        dataset = data.create_with_index(df_graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)

        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def Embed_generator():
    context = configs.Embed()
    dataset_files = data.get_directory_files(PATHS.cpg)

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)

        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")

        rows = []
        for _, row in cpg_dataset.iterrows():
            func_dicts = parse_to_functions(row.cpg, max_nodes=context.nodes_dim)
            for nodes in func_dicts:
                g = process.nodes_to_input(nodes, row.target, context.nodes_dim, mode="codebert")
                rows.append({"input": g, "target": row.target, "func": row.func})

        out_df = pd.DataFrame(rows, columns=["input", "target", "func"])
        print(f"[INFO] saving INPUT rows={len(out_df)}  file={file_name}_{FILES.input}")
        if len(out_df) == 0:
            print("[WARN] No functions parsed from this CPG chunk. Check parser/filter settings.")
        data.write(out_df, PATHS.input, f"{file_name}_{FILES.input}")


def train(model, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)
        y_pred = model(batch)
        model.zero_grad()
        batch.y = batch.y.squeeze().long()
        loss = criterion(y_pred, batch.y)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(
                epoch,
                (batch_idx + 1) * len(batch),
                len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1,
                100. * batch_idx / len(train_loader),
                loss.item()
            ))


def validate(model, val_loader):
    model.eval()
    test_loss = 0.0
    y_true, y_pred_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch.to(device)
            y_ = model(batch)
            batch.y = batch.y.squeeze().long()
            test_loss += F.cross_entropy(y_, batch.y).item()
            pred = y_.max(-1, keepdim=True)[1]
            y_true.extend(batch.y.cpu().numpy())
            y_pred_labels.extend(pred.cpu().numpy())

    test_loss /= len(val_loader)

    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Val)')
    plt.savefig('confusion_matrix_val.png')
    plt.close()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100
    ))
    return accuracy, precision, recall, f1


def _resolve_save_path(user_path: str | None, default_dir: str = "./saved_models", filename: str = "bertggcn.pth") -> Path:
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


if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-cpg', '--cpg', action='store_true', help='CPG generation task')
    parser.add_argument('-embed', '--embed', action='store_true', help='Embedding generation task')
    parser.add_argument('-mode', '--mode', default="train", help='train / test')
    parser.add_argument('-path', '--path', default=None, help='model path or dir')
    args = parser.parse_args()

    if args.cpg:
        CPG_generator()
    if args.embed:
        Embed_generator()

    # === データ読み込み・分割 ===
    context = configs.Process()
    input_dataset = loads(PATHS.input)
    train_ds, val_ds, test_ds = train_val_test_split(input_dataset, shuffle=context.shuffle)

    # === クラス不均衡対策 ===
    cnt = Counter(int(sample.y) for sample in train_ds)
    num0, num1 = cnt.get(0, 0), cnt.get(1, 0)
    N = max(1, num0 + num1)
    w0 = N / (2 * max(1, num0))
    w1 = N / (2 * max(1, num1))
    print(f"[ClassCount] num0={num0}, num1={num1} -> w0={w0:.4f}, w1={w1:.4f}")

    class_weights = torch.tensor([w0, w1], dtype=torch.float, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    sample_weights = [(w1 if int(sample.y) == 1 else w0) for sample in train_ds]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # === DataLoader 作成（PyGのDataLoaderを必ず使う） ===
    def build_loader(ds, batch_size, use_sampler=False, sampler_obj=None, shuffle=False):
        # ds.get_loader がある場合はそれを優先（互換のため）
        if hasattr(ds, "get_loader"):
            try:
                return ds.get_loader(
                    batch_size,
                    shuffle=(False if use_sampler else shuffle),
                    sampler=(sampler_obj if use_sampler else None)
                )
            except TypeError:
                pass
        # フォールバック：必ず GeoDataLoader を使う（PyG Data を正しくバッチ化）
        return GeoDataLoader(
            ds,
            batch_size=batch_size,
            sampler=(sampler_obj if use_sampler else None),
            shuffle=(False if use_sampler else shuffle)
        )

    train_loader = build_loader(train_ds, context.batch_size, use_sampler=False, sampler_obj=None, shuffle=True)
    val_loader   = build_loader(val_ds,   context.batch_size, use_sampler=False, sampler_obj=None, shuffle=False)
    test_loader  = build_loader(test_ds,  context.batch_size, use_sampler=False, sampler_obj=None, shuffle=False)

    print("Using device:", device)

    # === モデル・最適化器 ===
    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]

    SAVE_PATH = _resolve_save_path(args.path)

    if args.mode == "train":
        model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Bertggnn.learning_rate, weight_decay=Bertggnn.weight_decay)

        best_acc = 0.0
        NUM_EPOCHS = context.epochs
        for epoch in range(1, NUM_EPOCHS + 1):
            train(model, train_loader, optimizer, epoch, criterion)
            acc, precision, recall, f1 = validate(model, val_loader)
            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), str(SAVE_PATH))
            print("acc is: {:.4f}, best acc is {:.4f}".format(acc, best_acc))

    # === ベストモデルで test ===
    model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
    model_test.load_state_dict(torch.load(str(SAVE_PATH), map_location=device))
    accuracy, precision, recall, f1 = test(model_test, device, test_loader)
