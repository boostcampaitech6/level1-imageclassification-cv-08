import pandas as pd

df = pd.read_csv(r"C:\Users\haak1\Desktop\boostcamp\project1\train\train.csv")

id_list = []
label_list = []
for i in range(len(df["id"])):
    tmp = df["id"][i]
    id_list.append(tmp + "_incorrect_mask")
    label_list.append(2)
    id_list.append(tmp + "_mask1")
    label_list.append(0)
    id_list.append(tmp + "_mask2")
    label_list.append(0)
    id_list.append(tmp + "_mask3")
    label_list.append(0)
    id_list.append(tmp + "_mask4")
    label_list.append(0)
    id_list.append(tmp + "_mask5")
    label_list.append(0)
    id_list.append(tmp + "_normal")
    label_list.append(1)

df2 = pd.DataFrame({"id": id_list, "mask_label": label_list})
df2.to_csv(
    r"C:\Users\haak1\Desktop\boostcamp\project1\train\mask_train.csv", index=False
)
