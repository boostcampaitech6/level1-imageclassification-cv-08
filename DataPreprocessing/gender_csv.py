import pandas as pd

df = pd.read_csv(r"C:\Users\haak1\Desktop\boostcamp\project1\train\train.csv")

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
    if df["gender"][i] == "male":
        label_list += [0] * 7
    if df["gender"][i] == "female":
        label_list += [1] * 7

df2 = pd.DataFrame({"id": id_list, "gender_label": label_list})
# print(df2)
df2.to_csv(
    r"C:\Users\haak1\Desktop\boostcamp\project1\train\gender_train.csv", index=False
)
