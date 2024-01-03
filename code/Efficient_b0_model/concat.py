import pandas as pd

mask=pd.read_csv('/home/kangdg22/meta_Assignment/boostcamp/my_code/output/b3_cutmix/df_cutmix_mask.csv')
age=pd.read_csv('/home/kangdg22/meta_Assignment/boostcamp/my_code/output/b3_cutmix/df_cutmix_age.csv')
gender=pd.read_csv('/home/kangdg22/meta_Assignment/boostcamp/my_code/output/b3_cutmix/df_cutmix_gender.csv')

merged_df=pd.merge(mask, age, on='ImageID')
merged_df = pd.merge(merged_df, gender, on='ImageID')

def determine_label(row):
    if row['mask']==0 and row['age']==0 and row['gender']==0:
        return 0
    
    elif row['mask']==0 and row['age']==1 and row['gender']==0:
        return 1
    
    elif row['mask']==0 and row['age']==2 and row['gender']==0:
        return 2
    
    elif row['mask']==0 and row['age']==0 and row['gender']==1:
        return 3
    
    elif row['mask']==0 and row['age']==1 and row['gender']==1:
        return 4
    
    elif row['mask']==0 and row['age']==2 and row['gender']==1:
        return 5
    
    elif row['mask']==2 and row['age']==0 and row['gender']==0:
        return 6
    
    elif row['mask']==2 and row['age']==1 and row['gender']==0:
        return 7
    
    elif row['mask']==2 and row['age']==2 and row['gender']==0:
        return 8
    
    elif row['mask']==2 and row['age']==0 and row['gender']==1:
        return 9
    
    elif row['mask']==2 and row['age']==1 and row['gender']==1:
        return 10
    
    elif row['mask']==2 and row['age']==2 and row['gender']==1:
        return 11
    
    elif row['mask']==1 and row['age']==0 and row['gender']==0:
        return 12
    
    elif row['mask']==1 and row['age']==1 and row['gender']==0:
        return 13
    
    elif row['mask']==1 and row['age']==2 and row['gender']==0:
        return 14
    
    elif row['mask']==1 and row['age']==0 and row['gender']==1:
        return 15
    
    elif row['mask']==1 and row['age']==1 and row['gender']==1:
        return 16
    
    elif row['mask']==1 and row['age']==2 and row['gender']==1:
        return 17

merged_df['ans']=merged_df.apply(determine_label, axis=1)

print(merged_df)

merged_df=merged_df.drop(['mask', 'age', 'gender'], axis=1)

merged_df.to_csv('/result.csv')
