"""
Name:  Mohammed Saab
Email: mohammed.saab40@myhunter.cuny.edu
Resources: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
https://scikit-learn.org/stable/modules/clustering.html
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
and youtube videos 

"""

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

# Function to load data from a CSV file and clean it
def make_df(filename):

    # Read data from file
    df = pd.read_csv(filename)

    # Remove rows with missing values in specified columns
    required_columns = ['CREATE_DATE', 'INCIDENT_DATE', 'INCIDENT_TIME', 'Latitude', 'Longitude']

    # Filter rows that contain 'AMBULANCE' in 'TYP_DESC'
    type_desc_column = 'TYP_DESC' 
    df.dropna(subset=required_columns, inplace=True)
    df = df[df[type_desc_column].str.contains('AMBULANCE', na=False)]

    return df

# Function to add time-related features to the dataset
def add_date_time_features(df):
    date_column = 'INCIDENT_DATE'
    time_column = 'INCIDENT_TIME'
    
   # Convert date and time columns to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df[time_column] = pd.to_datetime(df[time_column], format='%H:%M:%S')

    # Calculate 'WEEK_DAY'
    df['WEEK_DAY'] = df[date_column].dt.dayofweek

    # Calculate 'INCIDENT_MIN' with fractional minutes
    df['INCIDENT_MIN'] = df[time_column].dt.hour * 60 + df[time_column].dt.minute + df[time_column].dt.second / 60

    return df


# Function to filter data based on the time of the incident
def filter_by_time(df, days=None, start_min=0, end_min=1439):

    # If days is None, default to all days of the week
    if days is None:
        days = list(range(7))

    # Filter the DataFrame for the specified days
    df_filtered = df[df['WEEK_DAY'].isin(days)]

    # Filter for the specified time range
    df_filtered = df_filtered[(df_filtered['INCIDENT_MIN'] >= start_min) & (df_filtered['INCIDENT_MIN'] <= end_min)]

    return df_filtered
# Function to perform KMeans clustering
def compute_kmeans(df, num_clusters=8, n_init='auto', random_state=1870):

    # Select only the latitude and longitude columns for clustering
    latitude_column = 'Latitude'
    longitude_column = 'Longitude'
    
     # Select only the latitude and longitude columns for clustering
    lat_long = df[[latitude_column, longitude_column]]

     # Determine the number of initializations
    if n_init == 'auto':
        n_init = 10 if num_clusters <= 8 else 2 * num_clusters

    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(lat_long)

    # Get the cluster centers and labels
    return kmeans.cluster_centers_, kmeans.labels_

# Function to perform Gaussian Mixture Model clustering
def compute_gmm(df, num_clusters=8, random_state=1870):

# Select only the latitude and longitude columns for clustering
    latitude_column = 'Latitude'
    longitude_column = 'Longitude'
    
     # Select only the latitude and longitude columns for clustering
    lat_long = df[[latitude_column, longitude_column]]

    # Create and fit the GaussianMixture model
    gmm = GaussianMixture(n_components=num_clusters, random_state=random_state)
    gmm.fit(lat_long)

    # Get the predicted labels
    return gmm.predict(lat_long)

# Function to perform Agglomerative clustering
def compute_agglom(df, num_clusters=8, linkage='ward'):

   # Select only the latitude and longitude columns for clustering
    latitude_column = 'Latitude'
    longitude_column = 'Longitude'
    
     # Select only the latitude and longitude columns for clustering
    lat_long = df[[latitude_column, longitude_column]]

     # Create and fit the AgglomerativeClustering model
    agglom = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
    agglom.fit(lat_long)

    # Get the predicted labels
    return agglom.labels_

# Function to perform Spectral clustering
def compute_spectral(df, num_clusters=8, affinity='rbf', random_state=1870):
       
   # Select only the latitude and longitude columns for clustering
    latitude_column = 'Latitude'
    longitude_column = 'Longitude'
    
     # Select only the latitude and longitude columns for clustering
    lat_long = df[[latitude_column, longitude_column]]

    # Create and fit the SpectralClustering model
    spectral = SpectralClustering(n_clusters=num_clusters, affinity=affinity, random_state=random_state)
    spectral.fit(lat_long)

    # Get the predicted labels
    return spectral.labels_

# Function to compute explained variance for different numbers of clusters
def compute_explained_variance(df, k_vals=None, random_state=1870):

    if k_vals is None:
        k_vals = [1, 2, 3, 4, 5]

    # Select only the latitude and longitude columns for clustering
    latitude_column = 'Latitude'
    longitude_column = 'Longitude'
    
     # Select only the latitude and longitude columns for clustering
    lat_long = df[[latitude_column, longitude_column]]

     # Calculate the sum of squared distances for each K
    sse = []
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(lat_long)
        sse.append(kmeans.inertia_)  # Inertia: Sum of squared distances of samples to their closest cluster center

    return sse


import pandas as pd

def test_add_date_time_features():
    
    # Create a sample DataFrame
    sample_data = {
        'INCIDENT_DATE': ['2021-07-04', '2021-07-05'],
        'INCIDENT_TIME': ['01:00:00', '02:30:00']
    }
    df = pd.DataFrame(sample_data)

    # Expected output
    expected_week_day = [6, 0] 
    expected_incident_min = [60, 150]  

    # Apply function
    df = add_date_time_features(df)

    # Check results
    if all(df['WEEK_DAY'] == expected_week_day) and all(df['INCIDENT_MIN'] == expected_incident_min):
        return True
    else:
        return False
    
def test_filter_by_time():

    # Create a sample DataFrame
    sample_data = {
        'WEEK_DAY': [0, 1, 2, 3, 4, 5, 6],
        'INCIDENT_MIN': [100, 200, 300, 400, 500, 600, 700]
    }
    df = pd.DataFrame(sample_data)

    # Apply filter_by_time function
    filtered_df = filter_by_time(df, days=[0, 1, 2], start_min=150, end_min=350)

    # Expected output: Should only include entries from Monday and Tuesday between 150 and 350 minutes
    expected_output = df[(df['WEEK_DAY'].isin([0, 1, 2])) & (df['INCIDENT_MIN'] >= 150) & (df['INCIDENT_MIN'] <= 350)]

    # Check if the filtered output matches the expected output
    return filtered_df.equals(expected_output)

