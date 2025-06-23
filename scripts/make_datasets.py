from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Замените на свои датасеты и сплиты
data_path_1 = "IlyaGusev/habr"
data_path_2 = "IlyaGusev/gazeta"
split_1 = "train"
split_2 = "train"

# Загружаем первый датасет и обрабатываем
print("Loading first dataset...")
ds1 = load_dataset(data_path_1, split=split_1)
df1 = ds1.to_pandas()

# Удаляем строки с коротким text_markdown
q95 = df1["text_markdown"].str.len().quantile(0.95)
df1_filtered = df1[df1["text_markdown"].str.len() >= q95].copy()
df1_filtered = df1_filtered[["title", "text_markdown"]].rename(columns={"text_markdown": "text"})

# Загружаем второй датасет
print("Loading second dataset...")
ds2 = load_dataset(data_path_2, split=split_2)
df2 = ds2.to_pandas()
df2 = df2[["title", "text"]]

# Объединяем два датафрейма
print("Concatenating datasets...")
combined_df = pd.concat([df1_filtered, df2], ignore_index=True)

# Разделение на train и test
print("Splitting into train and test sets...")
train_df, test_df = train_test_split(combined_df, test_size=0.05, random_state=42)

# Сохраняем в CSV
train_output_path = "data/train_dataset.csv"
test_output_path = "data/test_dataset.csv"
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)
print(f"Saved train dataset to {train_output_path}")
print(f"Saved test dataset to {test_output_path}")
