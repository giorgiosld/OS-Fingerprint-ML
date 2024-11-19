import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Read the CSV data
df = pd.read_csv("/home/data/meta_info_new.csv", header=None)

# The kernel version appears to be in column 4 (0-based index)
# Extract the version before the dash
def extract_version(version_string):
    return version_string.split('-')[0]

# Apply the extraction to the kernel version column
df['clean_version'] = df[4].apply(extract_version)

# Delete first row since it contains the world "Kernel Version"
df = df.iloc[1:]

# Count the occurrences of each version
version_counts = df['clean_version'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=version_counts.index, y=version_counts.values)

# Customize the plot
plt.title('Distribution of Ubuntu Kernel Versions')
plt.xlabel('Kernel Version')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot in the current directory
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ubuntu_version_distribution.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {plot_path}")

# Show the plot
plt.show()

# Print the numerical distribution
print("\nVersion Distribution:")
for version, count in version_counts.items():
    print(f"Version {version}: {count} instances")
