import pandas as pd

input_path = "./criteo_osrct/criteo_osrct_conversion_direct_alpha_2p0_complement.csv.gz"
output_path = "./criteo_osrct/criteo_osrct_conversion_direct_alpha_2p0_complement.csv"

df = pd.read_csv(input_path, compression="gzip")
print("shape:", df.shape)
print("columns:", list(df.columns))
print(df.head())

df.to_csv(output_path, index=False)

print("saved to:", output_path)