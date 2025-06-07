#Part of data inspection
#Part of data inspection by 202234883 Namjiwon
import os
import pandas as pd
import kagglehub

# 1. Download dataset using kagglehub
path = kagglehub.dataset_download("aaronschlegel/austin-animal-center-shelter-outcomes-and")
print("Path to dataset files:", path)

# 2. Get list of CSV files in the dataset directory
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

# 3. Check that there are at least two CSV files
if len(csv_files) >= 2:

 # 4. Select the second CSV file (index 1)
 selected_file = csv_files[1]
 file_path = os.path.join(path, selected_file)

 # 5. Load the selected file into a DataFrame
 df = pd.read_csv(file_path)
 print(f"\n[File Loaded: {selected_file}]")
 print("\n1. Dataset Shape:")
 print(df.shape) # Number of rows and columns
 print("\n2. Column Names:")
 print(df.columns.tolist()) # List of column headers
 print("\n3. Data Types:")
 print(df.dtypes) # Data type of each column
 print("\n4. Missing Values:")
 print(df.isnull().sum()) # Count of missing values per column
 print("\n5. Summary Statistics:")
 print(df.describe(include='all')) # Summary including categorical columns
 print("\n6. First 5 Rows:")
 print(df.head()) # Show first 5 rows
else:
 print("There are less than 2 CSV files in the dataset.")

#Part of data preparation by 202035325 Kimtaewan
import numpy as np
print("\n\n==============Start data preparation==============\n\n")

#Mapping unit to the days
unit_to_days = {
 "day": 1,
 "week": 7,
 "month": 30,
 "year": 365
}

#The function to change unit to days
def convert_age_to_days(age_str):
 if pd.isnull(age_str):
    return np.nan
 try:
    num, unit = age_str.lower().split()
    unit = unit.rstrip("s") # delete plural
    return int(num) * unit_to_days.get(unit, np.nan)
 except:
    return np.nan

#Convert 'age_upon_outcome' to' 'age_day'
df["age_days"] = df["age_upon_outcome"].apply(convert_age_to_days)
print("\nChecking the convert is right\n",df[["age_upon_outcome",
"age_days"]].head(10))
print(df.columns.tolist())

#Drop the feature which not necessary
drop_features = ["age_upon_outcome","animal_id", "date_of_birth", "datetime",
"monthyear", "name"]
df = df.drop(columns=drop_features) # drop!!
print("\n",df.columns)

#Check the distribution of major categorical variables
print("Animal Type:\n", df["animal_type"].value_counts(), "\n")
print("Sex upon Outcome:\n", df["sex_upon_outcome"].value_counts(), "\n")
print("Outcome Type:\n", df["outcome_type"].value_counts(), "\n")

#Divide the 'sex_upon_outcome' to 'sex' & 'fix_status'
def extract_sex(val):
 if pd.isnull(val) or val.lower() == 'unknown':
    return 'Unknown'
 return val.split()[-1]

def extract_fix(val):
 if pd.isnull(val) or val.lower() == 'unknown':
    return 'Unknown'
 fix = val.split()[0]
 if fix in ['Spayed', 'Neutered']:
    return 'Fixed'
 elif fix == 'Intact':
    return 'Intact'
 else:
    return 'Unknown'

#Create New feature
df["sex"] = df["sex_upon_outcome"].apply(extract_sex)
df["fix_status"] = df["sex_upon_outcome"].apply(extract_fix)
#Checking
print(df[["sex_upon_outcome", "sex", "fix_status"]].head(10))
#feature drop
df = df.drop(columns=["sex_upon_outcome"])

#Check the number of missing values
missing_counts = df.isnull().sum()
missing_columns = missing_counts[missing_counts > 0]
print("\nMissing Values:\n", missing_columns)

#Missing value handling
df = df[df["outcome_type"].notnull()] #Remove rows without labels
df["age_days"] = df["age_days"].fillna(df["age_days"].mean())
df["sex"] = df["sex"].fillna("Unknown")
df["fix_status"] = df["fix_status"].fillna("Unknown")

#Scaling group by animal type cause animal's max age&& min age may be different
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer


#Define Decimal Scaling
def decimal_scaling(arr):
 max_abs = np.max(np.abs(arr))
 if max_abs == 0:
    return arr
 j = len(str(int(max_abs)))
 return arr / (10 ** j)

#Apply One-Hot Encoding
# 1. Columns to be one-hot encoded
onehot_cols = ["animal_type", "sex", "fix_status", "outcome_type", "breed", "color"]

# 2. Handle missing values in string columns
df[onehot_cols] = df[onehot_cols].fillna("Unknown")

# 3. Copy the necessary columns for encoding
df_for_encoding = df.copy()

# 4. Apply one-hot encoding
df_encoded_one_hot = pd.get_dummies(df_for_encoding, columns=onehot_cols)

# 5. Check if all columns are numeric (for verification)
print("After one-hot encoding, all columns are numeric?:", all(np.issubdtype(dtype, np.number) for dtype in df_encoded_one_hot.dtypes))


#Initialize
scaler_cols = ["minmax", "standard", "robust", "maxabs", "normalizer", "decimal", "log"]
for col in scaler_cols:
 df_encoded_one_hot[f"age_days_{col}_by_type"] = np.nan

#Add Scaler for One-Hot Encoding
for group, sub_df in df.groupby("animal_type"):
 idx = sub_df.index
 arr = sub_df[["age_days"]].values

 df_encoded_one_hot.loc[idx, "age_days_minmax_by_type"] = MinMaxScaler().fit_transform(arr)
 df_encoded_one_hot.loc[idx, "age_days_standard_by_type"] = StandardScaler().fit_transform(arr)
 df_encoded_one_hot.loc[idx, "age_days_robust_by_type"] = RobustScaler().fit_transform(arr)
 df_encoded_one_hot.loc[idx, "age_days_maxabs_by_type"] = MaxAbsScaler().fit_transform(arr)
 df_encoded_one_hot.loc[idx, "age_days_normalizer_by_type"] = Normalizer().fit_transform(arr)
 df_encoded_one_hot.loc[idx, "age_days_decimal_by_type"] = decimal_scaling(arr).flatten()
 df_encoded_one_hot.loc[idx, "age_days_log_by_type"] = np.log1p(arr).flatten()

#Apply label Encoding
df_encoded_label = df.copy()

#Initailize
for col in scaler_cols:
 df_encoded_label[f"age_days_{col}_by_type"] = np.nan

#Scaler for label Encoding
for group, sub_df in df_encoded_label.groupby("animal_type"):
 idx = sub_df.index
 arr = sub_df[["age_days"]].values

 df_encoded_label.loc[idx, "age_days_minmax_by_type"] = MinMaxScaler().fit_transform(arr)
 df_encoded_label.loc[idx, "age_days_standard_by_type"] = StandardScaler().fit_transform(arr)
 df_encoded_label.loc[idx, "age_days_robust_by_type"] = RobustScaler().fit_transform(arr)
 df_encoded_label.loc[idx, "age_days_maxabs_by_type"] = MaxAbsScaler().fit_transform(arr)
 df_encoded_label.loc[idx, "age_days_normalizer_by_type"] = Normalizer().fit_transform(arr)
 df_encoded_label.loc[idx, "age_days_decimal_by_type"] = decimal_scaling(arr).flatten()
 df_encoded_label.loc[idx, "age_days_log_by_type"] = np.log1p(arr).flatten()

label_cols = ["animal_type", "sex", "fix_status", "outcome_type", "breed", "color"]
for col in label_cols:
 le = LabelEncoder()
 df_encoded_label[col] = le.fit_transform(df_encoded_label[col])

#Final checking
print("\n1.One-Hot Encoding: \n", df_encoded_one_hot,"\n")
print(df_encoded_one_hot.shape)
print("\n2. Column Names:")
print(df_encoded_one_hot.columns.tolist())
print("\n3. Data Types:")
print(df_encoded_one_hot.dtypes)
print("\n4. Missing Values:")
print(df_encoded_one_hot.isnull().sum())
print("\n5. Summary Statistics:")
print(df_encoded_one_hot.describe(include='all'))
print("\n6. First 5 Rows:")
print(df_encoded_one_hot.head())
print("\n\n1.Label Encoding:\n", df_encoded_label,"\n")
print(df_encoded_label.shape)
print("\n2. Column Names:")
print(df_encoded_label.columns.tolist())
print("\n3. Data Types:")
print(df_encoded_label.dtypes)
print("\n4. Missing Values:")
print(df_encoded_label.isnull().sum())
print("\n5. Summary Statistics:")
print(df_encoded_label.describe(include='all'))
print("\n6. First 5 Rows:")
print(df_encoded_label.head())


# Part of Analysis Algorithms by 202234911 유연이
# KNN (Classification) and K-Means (Clustering) with various parameter combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("\n\n==============Start Analysis Algorithms==============\n\n")

# 1. Define scaling methods list
scaling_methods = ['original', 'minmax', 'standard', 'robust', 'maxabs', 'normalizer', 'decimal', 'log']

# 2. Define model creation functions
def create_knn_model(n_neighbors=5, metric='euclidean', weights='uniform'):
    """
    KNN model creation function
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)

def create_kmeans_model(n_clusters=3, init='k-means++', random_state=42):
    """
    K-Means model creation function
    """
    return KMeans(n_clusters=n_clusters, init=init, random_state=random_state, n_init=10)

# 3. Define parameter combinations
    # n_neighbors: number of closest neighbors to consider for classification
    # metric: method for measuring distance between data points
    # weights: neighbor weighting method (uniform: all neighbors equal, distance: closer neighbors given higher weight)
knn_params = [
    {"n_neighbors": 3, "metric": "euclidean", "weights": "uniform"},
    {"n_neighbors": 5, "metric": "manhattan", "weights": "distance"},
    {"n_neighbors": 7, "metric": "minkowski", "weights": "uniform"},
    {"n_neighbors": 9, "metric": "euclidean", "weights": "distance"},
    {"n_neighbors": 5, "metric": "manhattan", "weights": "uniform"},
]

    # n_clusters: number of clusters (groups) to divide the data into
    # init: method for initializing center points (k-means++: distributed initialization, random: random initialization)
kmeans_params = [
    {"n_clusters": 3, "init": "k-means++"},
    {"n_clusters": 4, "init": "random"},
    {"n_clusters": 5, "init": "k-means++"},
    {"n_clusters": 6, "init": "random"},
    {"n_clusters": 7, "init": "k-means++"},
]

print("Model parameter combinations defined")
print(f"- KNN parameter combinations: {len(knn_params)}")
print(f"- K-Means parameter combinations: {len(kmeans_params)}")

# 4. Prepare one-hot encoded datasets
print("\nPreparing one-hot encoded datasets...")
# Prepare datasets with different scaling versions for one-hot encoded data
onehot_datasets = {}  # {scaling_method: (X, y)}

for scale_method in scaling_methods:
    # Copy original data
    df_temp = df_encoded_one_hot.copy()
    
    # Change age_days to the selected scaling method, since age_days is the only numerical variable
    if scale_method != 'original':
        scale_col = f'age_days_{scale_method}_by_type'
        if scale_col in df_temp.columns:
            df_temp['age_days'] = df_temp[scale_col]
    
    # Remove unused scaling columns, since we have multiple versions of the same information (age) in the dataframe
    cols_to_drop = [col for col in df_temp.columns if ('age_days_' in col and col != 'age_days')]
    df_temp = df_temp.drop(columns=cols_to_drop)
    
    # Find outcome_type columns
    outcome_cols = [col for col in df_temp.columns if 'outcome_type_' in col]
    
    if len(outcome_cols) > 0:
        # Set feature variables - all columns except outcome_type columns
        X = df_temp.drop(columns=outcome_cols).values
        
        # Prepare target variable for multi-class classification
        # Restore original classes from one-hot encoded outcome_type columns
        # (Find the index with 1 in each row)
        outcomes_matrix = df_temp[outcome_cols].values  # One-hot encoded outcome columns
        y_multiclass = np.argmax(outcomes_matrix, axis=1)  # Index with 1 in each row
        
        # Save class names (for mapping index → actual class name)
        class_names = [col.replace('outcome_type_', '') for col in outcome_cols]
        
        # Save dataset
        onehot_datasets[scale_method] = (X, y_multiclass, class_names)
        print(f"- One-hot encoding + {scale_method} scaling: X shape={X.shape}, y shape={y_multiclass.shape}, classes={len(class_names)}")
        print(f"  Classes: {class_names}")

# 5. Prepare label encoded datasets
print("\nPreparing label encoded datasets...")
# Prepare datasets with different scaling versions for label encoded data
label_datasets = {}  # {scaling_method: (X, y)}

for scale_method in scaling_methods:
    # Copy original data
    df_temp = df_encoded_label.copy()
    
    # Change age_days to the selected scaling method
    if scale_method != 'original':
        scale_col = f'age_days_{scale_method}_by_type'
        if scale_col in df_temp.columns:
            df_temp['age_days'] = df_temp[scale_col]
    
    # Remove unused scaling columns
    cols_to_drop = [col for col in df_temp.columns if ('age_days_' in col and col != 'age_days')]
    df_temp = df_temp.drop(columns=cols_to_drop)
    
    # Set target and feature variables
    y = df_temp['outcome_type'].values
    X = df_temp.drop(columns=['outcome_type']).values
    
    # Find actual class names corresponding to label encoded outcome_type
    # (In the current data, LabelEncoder has already been applied so original mapping information is lost)
    # Use original df to get unique class names
    outcome_classes = df['outcome_type'].unique().tolist()
    
    # Save dataset
    label_datasets[scale_method] = (X, y, outcome_classes)
    print(f"- Label encoding + {scale_method} scaling: X shape={X.shape}, y shape={y.shape}, classes={len(outcome_classes)}")
    print(f"  Classes: {outcome_classes}")

# 6. Create models
print("\nCreating models...")
# Create model objects and save preprocessing+model+parameter combination information

# List to store all model combinations
all_model_combinations = []

# Calculate total number of model combinations
total_combinations = len(scaling_methods) * (len(knn_params) + len(kmeans_params)) * 2  # 2 is for encoding methods
print(f"Total model combinations to create: {total_combinations}")

# 6.1 Create models with one-hot encoded data
print("\nCreating models with one-hot encoded data...")
for scale_method, (X, y, class_names) in onehot_datasets.items():
    # KNN models
    for i, params in enumerate(knn_params):
        model_name = f"KNN_{i+1}"
        
        # Save model information
        model_info = {
            "encoding": "one-hot",
            "scaling": scale_method,
            "model_type": "KNN",
            "model_name": model_name,
            "parameters": params,
            "model": create_knn_model(**params),
            "X": X,
            "y": y,
            "classes": class_names,  # Class names
            "prediction_type": "multi-class",  # Multi-class classification
            "id": f"OneHot_{scale_method}_KNN_{i+1}"
        }
        all_model_combinations.append(model_info)
        print(f"- Model created: OneHot_{scale_method}_{model_name}")
    
    # K-Means models
    for i, params in enumerate(kmeans_params):
        model_name = f"KMeans_{i+1}"
        
        # Save model information
        model_info = {
            "encoding": "one-hot",
            "scaling": scale_method,
            "model_type": "K-Means",
            "model_name": model_name,
            "parameters": params,
            "model": create_kmeans_model(**params),
            "X": X,
            "y": None,  # No target needed for clustering
            "classes": class_names,  # Class names (for cluster analysis)
            "prediction_type": "clustering",
            "id": f"OneHot_{scale_method}_KMeans_{i+1}"
        }
        all_model_combinations.append(model_info)
        print(f"- Model created: OneHot_{scale_method}_{model_name}")

# 6.2 Create models with label encoded data
print("\nCreating models with label encoded data...")
for scale_method, (X, y, class_names) in label_datasets.items():
    # KNN models
    for i, params in enumerate(knn_params):
        model_name = f"KNN_{i+1}"
        
        # Save model information
        model_info = {
            "encoding": "label",
            "scaling": scale_method,
            "model_type": "KNN",
            "model_name": model_name,
            "parameters": params,
            "model": create_knn_model(**params),
            "X": X,
            "y": y,
            "classes": class_names,  # Class names
            "prediction_type": "multi-class",  # Multi-class classification
            "id": f"Label_{scale_method}_KNN_{i+1}"
        }
        all_model_combinations.append(model_info)
        print(f"- Model created: Label_{scale_method}_{model_name}")
    
    # K-Means models
    for i, params in enumerate(kmeans_params):
        model_name = f"KMeans_{i+1}"
        
        # Save model information
        model_info = {
            "encoding": "label",
            "scaling": scale_method,
            "model_type": "K-Means",
            "model_name": model_name,
            "parameters": params,
            "model": create_kmeans_model(**params),
            "X": X,
            "y": None,  # No target needed for clustering
            "classes": class_names,  # Class names (for cluster analysis)
            "prediction_type": "clustering",
            "id": f"Label_{scale_method}_KMeans_{i+1}"
        }
        all_model_combinations.append(model_info)
        print(f"- Model created: Label_{scale_method}_{model_name}")

# 7. Model statistics and summary
print("\n\n==============Model Creation Complete==============")
print(f"Total {len(all_model_combinations)} model combinations created")

# Model statistics
knn_count = sum(1 for model in all_model_combinations if model["model_type"] == "KNN")
kmeans_count = sum(1 for model in all_model_combinations if model["model_type"] == "K-Means")

print(f"- KNN models: {knn_count}")
print(f"- K-Means models: {kmeans_count}")

# Statistics by encoding
onehot_count = sum(1 for model in all_model_combinations if model["encoding"] == "one-hot")
label_count = sum(1 for model in all_model_combinations if model["encoding"] == "label")

print(f"- One-hot encoded models: {onehot_count}")
print(f"- Label encoded models: {label_count}")

# Statistics by scaling method
print("\nModels by scaling method:")
for scale in scaling_methods:
    count = sum(1 for model in all_model_combinations if model["scaling"] == scale)
    print(f"- {scale}: {count}")

# 8. Output example model
if all_model_combinations:
    model = all_model_combinations[0]
    print("\nFirst model example:")
    print(f"- ID: {model['id']}")
    print(f"- Encoding: {model['encoding']}")
    print(f"- Scaling: {model['scaling']}")
    print(f"- Model type: {model['model_type']}")
    print(f"- Parameters: {model['parameters']}")
    print(f"- X shape: {model['X'].shape}")
    if model['y'] is not None:
        print(f"- y shape: {model['y'].shape}")
        print(f"- Prediction type: {model['prediction_type']}")
        print(f"- Number of classes: {len(model['classes'])}")
    else:
        print("- y: None (clustering model)")



#data for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array
import numpy as np

def is_numeric_array(X):
    try:
        check_array(X, dtype=np.float64)
        return True
    except:
        return False

print("\n\n============== Model Evaluation (First 20 Models Only) ==============")

knn_results = []
kmeans_results = []

# Evaluate only the first 20 models
for model_info in all_model_combinations[:20]:
    X = model_info["X"]
    model = model_info["model"]

    if not is_numeric_array(X):
        continue

    if model_info["model_type"] == "KNN":
        y = model_info["y"]
        try:
            scores = cross_val_score(model, X, y, cv=2, n_jobs=-1)
            mean_score = scores.mean()
            knn_results.append((model_info["id"], mean_score))
        except:
            continue

    elif model_info["model_type"] == "K-Means":
        try:
            X_sample = X[:5000]  # Optional sample limit
            X_pca = PCA(n_components=100).fit_transform(X_sample)
            labels = model.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            kmeans_results.append((model_info["id"], score))
        except:
            continue

# Sort and print top 5 for each model type
top5_knn = sorted(knn_results, key=lambda x: x[1], reverse=True)[:5]
top5_kmeans = sorted(kmeans_results, key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 KNN Models (by Mean Accuracy):")
for model_id, score in top5_knn:
    print(f"- {model_id}: {score:.4f}")

print("\nTop 5 K-Means Models (by Silhouette Score):")
for model_id, score in top5_kmeans:
    print(f"- {model_id}: {score:.4f}")



'''
Explanation (key variables for evaluation):
1. all_model_combinations: List containing all model combinations
2. Each model information includes:
   - encoding: Encoding method used (one-hot or label)
   - scaling: Scaling method used
   - model_type: Type of model used (KNN or K-Means)
   - model_name: Unique identifier for the model
   - parameters: Model parameter information
   - model: Model object (not yet trained)
   - X: Feature data for the model
   - y: Target data for the model (None for clustering)
   - classes: Class names (Adoption, Transfer, Return to Owner, etc.)
   - prediction_type: multi-class or clustering
   - id: Unique identifier for the combination
'''