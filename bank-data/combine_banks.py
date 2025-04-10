import pandas as pd

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
ocbc_df = pd.read_csv('ocbc/ocbc_data.csv')
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

# Save to CSV
combined_df.to_csv('combined_banks.csv', index=False)
print("Combined CSV created")
