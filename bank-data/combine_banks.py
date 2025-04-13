import pandas as pd
from collections import defaultdict

# DBS data
dbs_df = pd.read_csv('dbs/dbs_data.csv', encoding='latin1')
dbs_df = dbs_df.rename(columns={
    'URL': 'url',
    'Title': 'title',
    'Subtitle': 'subtitle',
    'Heading': 'header',
    'Information': 'text',
    'Metadata': 'tag'
})
dbs_df['bank'] = 'dbs'

# OCBC data
ocbc_df = pd.read_csv('ocbc/ocbc_data.csv', encoding='latin1')
ocbc_df = ocbc_df.rename(columns={
    'url': 'url',
    'title': 'title',
    'subtitle': 'subtitle',
    'subheader': 'header',
    'text': 'text',
    'tag': 'tag'
})
ocbc_df['bank'] = 'ocbc'

# Reorder columns
final_columns = ['bank', 'url', 'title', 'subtitle', 'header', 'text', 'tag']
dbs_df = dbs_df[final_columns]
ocbc_df = ocbc_df[final_columns]

# Combine the banks
combined_df = pd.concat([dbs_df, ocbc_df], ignore_index=True)
combined_df['original_order'] = combined_df.index

# Function to combine text per group
def combine_texts(df):
    df = df.sort_values('original_order')
    parts = []
    for _, row in df.iterrows():
        if pd.notna(row['header']) and str(row['header']).strip() != "":
            parts.append(str(row['header']).strip())
            parts.append(str(row['text']).strip())
        else:
            parts.append(str(row['text']).strip())
    combined = "\n".join(parts)
    return pd.DataFrame({'text': [combined]})

grouped = combined_df.groupby(['bank', 'url', 'title', 'subtitle'], as_index=False)

agg_df = grouped.agg({
    'tag': 'first',
    'original_order': 'min'
})

combined_texts = grouped.apply(combine_texts).reset_index(drop=True)
print(combined_texts.head())

final_df = pd.concat([agg_df, combined_texts], axis=1)

# Rearrange columns if needed
final_columns = ['bank', 'url', 'title', 'subtitle', 'text', 'tag']
final_df = final_df[final_columns]

# Save to CSV
final_df.to_csv('combined_banks.csv', index=False)
print("Aggregated and combined CSV created")
