import configs
import utils.data as data
import utils.process as process
import shutil
import gc
import pandas as pd
import utils.process as process
import utils.functions.cpg_dir as cpg

from utils.functions.cpg import parse_to_functions
from utils.process.embeddings import nodes_to_input


PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    result = dataset.loc[dataset['project'] == "qemu"] #select data from only qemu project
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    result = result.head(1000) # For debug, only 200 samples is extracted.
    
    return result

def CPG_generator():
    """
    Generates Code Property Graph (CPG) datasets from raw data.
    """

    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw) #DataFrame

    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    #Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)

    #Create CPG with graphs json files
    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if not graphs:
            print(f"Dataset chunk {s} not processed.")
            continue

        # graphs: List[Tuple[int, dict]] -> DataFrame へ
        df_graphs = pd.DataFrame(graphs, columns=["Index", "cpg"])
        df_graphs["Index"] = df_graphs["Index"].astype(int)

        # ここで初めて create_with_index に渡す
        dataset = data.create_with_index(df_graphs, ["Index", "cpg"])

        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writting cpg dataset chunk {s}.")
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

        out_df = pd.DataFrame(rows, columns=["input","target","func"])
        print(f"[INFO] saving INPUT rows={len(out_df)}  file={file_name}_{FILES.input}")
        if len(out_df) == 0:
            print("[WARN] No functions parsed from this CPG chunk. Check parser/filter settings.")
        data.write(out_df, PATHS.input, f"{file_name}_{FILES.input}")
