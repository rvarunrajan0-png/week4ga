import pandas as pd, numpy as np
df = pd.read_csv("data/v2.csv")
dup = df.sample(n=max(5,int(len(df)*0.1)), replace=True, random_state=42).reset_index(drop=True)
num_cols = df.select_dtypes(include='number').columns
for c in num_cols:
    dup[c] = dup[c] + np.random.normal(scale=0.01, size=len(dup))
df_aug = pd.concat([df, dup]).reset_index(drop=True)
df_aug.to_csv("data/v2.csv", index=False)
print("Wrote data/v2.csv")
