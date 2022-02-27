import sys, os, copy, json
import pandas as pd
from upper_bound import read_data

topic = sys.argv[1]

# seed,topic,comp,month,method,k,val,test,summ1,summ2,hyps
# bow = pd.read_csv("res/cbc_ideology_%s.csv"%topic).query("k == 4")
# bow["feats"] = "bow"
# emb = pd.read_csv("res/cbc3_ideology_%s.csv"%topic).query("k == 4")
# emb["feats"] = pd.read_csv("res/cbc_ideology_%s.csv"%topic).query("k == 4")
# bow["feats"]
cbc = pd.read_csv("res/cbc_ideology_%s.csv"%topic).query("k == 4")

feats_map = {"emb_exp_greedy-diff": "sbert", "tok_bow_exp_greedy-diff": "bow"}
print(cbc.method.values)
cbc["feats"] = [feats_map[m] for m in cbc.method.values]

cbc = cbc.sort_values('test', ascending=False).drop_duplicates(['comp','month', 'feats'])[["comp", "month", "feats", "test", "summ1", "summ2"]]


data = read_data(topic, False)

op = open("res/summs_%s.json"%topic, "w")

for row in cbc.T.to_dict().values():
    summ1 = row["summ1"].split(";")
    summ2 = row["summ2"].split(";")
    label1, label2 = row["comp"].split("_")
    data1 = data.query("id == @summ1")
    data2 = data.query("id == @summ2")

    # print(label1, label2)
    res = copy.copy(row)
    assert set(data1.ideology.values) ==  { label1 } or set(data1.ideology.values) ==  { label2 }
    assert set(data2.ideology.values) ==  { label2 } or set(data2.ideology.values) ==  { label1 }

    row["summ1"] = data1.title.values.tolist()
    row["summ2"] = data2.title.values.tolist()
    op.write(json.dumps(row) + "\n")

op.close()