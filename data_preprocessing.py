import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/codespace/.cache/kagglehub/datasets/jerryqu/reddit-conversations/versions/1/casual_data_windows.csv")
df.dropna(inplace=True)

print(df.head)
print(df.columns)

#Strip removes spaces at the beginning and at the end of the string
def preprocess_text(text):
    return text.strip()

for col in ['0','1','2']:
    df[col] = df[col].apply(preprocess_text)


print(df.head)




