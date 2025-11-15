# Graduation_Project
Please change the amound of data to use for train by adjusting the number of "result = result.head(1000)"

Preparing CPG and embed
python run.py -cpg -embed -mode



training:
python3 run.py -mode train -path ./data/model/


test:
python3 run.py -mode test -path ./data/model/bertggcn.pth



Trainingの後に以下のように一番良いパラメータが表示されますので、一番上のモデルを使ってください。
===== SWEEP SUMMARY (sorted by best_f1) =====
batch=16, lr=0.0001, best_f1=0.1848 (epoch 12), model=data/model/bert_rgcn_bs16_lr1e-04.pth
batch=32, lr=3e-05, best_f1=0.1817 (epoch 14), model=data/model/bert_rgcn_bs32_lr3e-05.pth
batch=8, lr=0.0001, best_f1=0.1686 (epoch 9), model=data/model/bert_rgcn_bs8_lr1e-04.pth
batch=32, lr=1e-05, best_f1=0.1436 (epoch 4), model=data/model/bert_rgcn_bs32_lr1e-05.pth
batch=16, lr=3e-05, best_f1=0.1369 (epoch 3), model=data/model/bert_rgcn_bs16_lr3e-05.pth
batch=8, lr=1e-05, best_f1=0.1237 (epoch 1), model=data/model/bert_rgcn_bs8_lr1e-05.pth
batch=8, lr=5e-05, best_f1=0.1230 (epoch 4), model=data/model/bert_rgcn_bs8_lr5e-05.pth
batch=16, lr=1e-05, best_f1=0.1171 (epoch 1), model=data/model/bert_rgcn_bs16_lr1e-05.pth
batch=16, lr=5e-05, best_f1=0.1038 (epoch 1), model=data/model/bert_rgcn_bs16_lr5e-05.pth
batch=32, lr=5e-05, best_f1=0.1010 (epoch 1), model=data/model/bert_rgcn_bs32_lr5e-05.pth
batch=32, lr=0.0001, best_f1=0.0334 (epoch 4), model=data/model/bert_rgcn_bs32_lr1e-04.pth
batch=8, lr=3e-05, best_f1=0.0063 (epoch 1), model=data/model/bert_rgcn_bs8_lr3e-05.pth

この例の場合は一番上のbert_rgcn_bs16_lr1e-04.pthが一番いいモデルなのでテストのときは以下のように実行します。
python3 run.py -mode test -path ./data/model/bert_rgcn_bs16_lr1e-04.pth

