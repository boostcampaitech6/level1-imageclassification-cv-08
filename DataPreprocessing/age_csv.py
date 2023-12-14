import pandas as pd

df = pd.read_csv("../train/train.csv")

id_list = []
label_list = []
for i in range(len(df["id"])):
    tmp = df["id"][i]
    id_list.append(tmp + "_incorrect_mask")
    id_list.append(tmp + "_mask1")
    id_list.append(tmp + "_mask2")
    id_list.append(tmp + "_mask3")
    id_list.append(tmp + "_mask4")
    id_list.append(tmp + "_mask5")
    id_list.append(tmp + "_normal")
    if int(df["age"][i]) < 30:
        label_list += [0] * 7
    if 30 <= int(df["age"][i]) < 60:
        label_list += [1] * 7
    if 60 <= int(df["age"][i]):
        label_list += [2] * 7

df2 = pd.DataFrame({"id": id_list, "age_label": label_list})

df2.to_csv("../age_train.csv", index=True)
