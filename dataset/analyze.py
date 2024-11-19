import pandas as pd

def analyze_ubuntu_versions(file_path):
    df = pd.read_csv(file_path)
    version_counts = df['label'].value_counts()
    ubuntu_versions = ['ubuntu,3.13.0', 'ubuntu,4.4.0', 'ubuntu,4.15.0', 'ubuntu,5.4.0']
    filtered_counts = {version: version_counts.get(version, 0) for version in ubuntu_versions}
    return filtered_counts

# Read data and analyze
data = pd.read_csv('dataset_200_full.csv')
results = analyze_ubuntu_versions('dataset_200_full.csv')

# Display results
for version, count in results.items():
    print(f"{version}: {count}")
