from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, completeness_score, davies_bouldin_score, f1_score, homogeneity_score, precision_score, recall_score, roc_auc_score, silhouette_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



# Function to load data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to display the data table
def display_data_table(data, target_variable):
    columns = list(data.columns)
    columns.append(columns.pop(columns.index(target_variable)))
    data = data[columns]
    st.write(data.head())

# PCA Visualization function
def pca_visualization(data, target_variable=None, all_features=False):
    data.columns = data.columns.str.strip()
    features = data.columns.drop(target_variable) if target_variable else data.columns
    numeric_features = data[features].select_dtypes(include=np.number)
    categorical_features = data[features].select_dtypes(exclude=np.number)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features.columns),
            ('cat', categorical_transformer, categorical_features.columns)])
    
    X = preprocessor.fit_transform(data[features])
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    if target_variable:
        principal_df['target'] = data[target_variable]
        targets = principal_df['target'].unique()

        colors = sns.color_palette("tab20", len(targets)) if len(targets) <= 20 else sns.color_palette("tab20b", len(targets))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('PC1', fontsize=15)
        ax.set_ylabel('PC2', fontsize=15)
        ax.set_title('2 Component PCA', fontsize=20)

        for target_value, color in zip(targets, colors):
            indices_to_keep = principal_df['target'] == target_value
            ax.scatter(principal_df.loc[indices_to_keep, 'PC1'],
                       principal_df.loc[indices_to_keep, 'PC2'],
                       c=[color], s=50, label=target_value)

        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('PC1', fontsize=15)
        ax.set_ylabel('PC2', fontsize=15)
        ax.set_title('2 Component PCA', fontsize=20)
        ax.scatter(principal_df['PC1'], principal_df['PC2'], c='blue', s=50, label='Data')

    ax.grid()
    st.pyplot(fig)

# t-SNE Visualization function
def t_sne_visualization(data, target_variable=None, perplexity=30, learning_rate=200):
    data.columns = data.columns.str.strip()
    features = data.columns.drop(target_variable) if target_variable else data.columns
    numeric_features = data[features].select_dtypes(include=np.number)
    categorical_features = data[features].select_dtypes(exclude=np.number)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features.columns),
            ('cat', categorical_transformer, categorical_features.columns)])

    X = preprocessor.fit_transform(data[features])

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    tsne_components = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'])

    if target_variable:
        tsne_df['target'] = data[target_variable]
        targets = tsne_df['target'].unique()
        colors = sns.color_palette("tab10", len(targets))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('t-SNE1', fontsize=15)
        ax.set_ylabel('t-SNE2', fontsize=15)
        ax.set_title('2 Component t-SNE', fontsize=20)

        for target, color in zip(targets, colors):
            indices_to_keep = tsne_df['target'] == target
            ax.scatter(tsne_df.loc[indices_to_keep, 't-SNE1'],
                       tsne_df.loc[indices_to_keep, 't-SNE2'],
                       c=[color], s=50, label=target)

        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('t-SNE1', fontsize=15)
        ax.set_ylabel('t-SNE2', fontsize=15)
        ax.set_title('2 Component t-SNE', fontsize=20)
        ax.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c='blue', s=50, label='Data')

    ax.grid()
    st.pyplot(fig)

# Function to split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Function to select a classifier
def select_classifier():
    return RandomForestClassifier()



# Function to encode categorical y
def encode_categorical_y(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded
#
def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    y_train_encoded, y_test_encoded = encode_categorical_y(y_train, y_test)

    classifier.fit(X_train, y_train_encoded)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='macro')
    recall = recall_score(y_test_encoded, y_pred, average='macro')
    f1 = f1_score(y_test_encoded, y_pred, average='macro')
   

    try:
        roc_auc = roc_auc_score(y_test_encoded, y_pred, average='macro', multi_class='ovo')
    except ValueError:
        roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        
    }
def perform_clustering(algorithm, X, **kwargs):#
    if algorithm == "KMeans":
        model = KMeans(**kwargs) #
    else:
        model = AgglomerativeClustering(**kwargs)  #
    clusters = model.fit_predict(X)
    return model, clusters


# Function to evaluate clustering
def evaluate_clustering(X, clusters, y_true):
    silhouette = silhouette_score(X, clusters)
    homogeneity = homogeneity_score(y_true, clusters)
    completeness = completeness_score(y_true, clusters)
    rand_index = adjusted_rand_score(y_true, clusters)
    
    try:
        calinski = calinski_harabasz_score(X, clusters)
    except ValueError:
        calinski = "Not defined for one cluster"
    try:
        davies = davies_bouldin_score(X, clusters)
    except ValueError:
        davies = "Not defined for one cluster"
    
    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "rand_index": rand_index
    }

def display_clustering_results(results):
   
    st.write(f"- Silhouette Score: {results['silhouette_score']:.4f}")
    st.write(f"- Calinski-Harabasz Score: {results['calinski_harabasz_score']}")
    st.write(f"- Davies-Bouldin Score: {results['davies_bouldin_score']}")
    st.write(f"- Homogeneity: {results['homogeneity']:.4f}")
    st.write(f"- Completeness: {results['completeness']:.4f}")
    st.write(f"- Adjusted Rand Index: {results['rand_index']:.4f}")
    


# Function to display classification results
def display_classification_results(results):
    st.write(f"Accuracy: {results['accuracy']:.4f}")
    st.write(f"Precision (Macro Average): {results['precision']:.4f}")
    st.write(f"Recall (Macro Average): {results['recall']:.4f}")
    st.write(f"F1-Score (Macro Average): {results['f1']:.4f}")
    if results['roc_auc']:
        st.write(f"ROC AUC Score (Macro Average, OVO): {results['roc_auc']:.4f}")
    
# Function to display clustering results



def feature_visualization(data):
    st.subheader("Feature visualization")
    st.write("Select the feature to visualize:")
    feature = st.selectbox("Feature", options=data.columns[:-1], key="feature_selection_selectbox")
    target_variable = data.columns[-1]

    if feature:
        if data[feature].dtype == 'object':
            # Count Plot for categorical feature
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=data, x=feature, ax=ax)
            ax.set_title(f"Count Plot for Categorical Feature {feature}")
            st.pyplot(fig)
        else:
            # Density Plot for numerical feature
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.kdeplot(data[feature], fill=True, ax=ax)
            ax.set_title(f"Density Plot for Numerical Feature {feature}")
            st.pyplot(fig)

            # Dot Plot for numerical feature
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.stripplot(x=data[target_variable], y=data[feature], ax=ax)
            ax.set_title(f"Dot Plot for Numerical Feature {feature} by {target_variable}")
            st.pyplot(fig)
            
def compare_classifiers(X_train, y_train, X_test, y_test):
    classifiers = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
    }

    results = {}
    for name, clf in classifiers.items():
        results[name] = train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test)

    best_classifier = max(results, key=lambda k: results[k]["accuracy"])
    st.subheader(f"Best Classifier: {best_classifier}")
    display_classification_results(results[best_classifier])






def compare_clustering_algorithms(X, y_true):
    algorithms = {
        "KMeans": KMeans(random_state=42),
        "Agglomerative Clustering": AgglomerativeClustering()
    }

    results = {}
    for name, alg in algorithms.items():
        model, clusters = perform_clustering(name, X, n_clusters=3)  # Use 3 clusters for comparison
        results[name] = evaluate_clustering(X, clusters, y_true)

    best_algorithm = max(results, key=lambda k: results[k]["silhouette_score"])
    st.subheader(f"Best Clustering Algorithm: {best_algorithm}")
    display_clustering_results(results[best_algorithm])





page_bg_img =st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: black; 
        color: silver;
    }}

    [data-testid="stAppViewContainer"] .main {{
        background-color: black;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background-color: black; 
        color: silver;
        border-right: 3px solid white; 
    }}

    [data-testid="stHeader"] {{
        background-color: black; 
        color: silver; 
    }}
    
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}

    
    </style>
    """,
    unsafe_allow_html=True,
)


## Streamlit App
st.title("Web App για αξιολόγιση και visualize αλγορίθμων μηχανικής μάθησης")



# Sidebar for data file upload
uploaded_file = st.sidebar.file_uploader("Ανέβασε το CSV file", type=["csv"])

if uploaded_file:
    if uploaded_file.size == 0:
        st.error("Παρακαλώ ανέβασε ένα αρχείο CSV")
    else:
        data = load_data(uploaded_file)

        if data is not None:
            st.write("### Dataset")
            st.write(data.head())

            # User selects the target variable
            target_variable = st.sidebar.selectbox("Επέλεξε την μεταβλητή στόχο", options=data.columns, key="target_variable_selectbox")

            # Display the updated data table with the target variable moved to the last position
            display_data_table(data, target_variable)

            tabs = st.tabs(["PCA", "t-SNE", "Feature Visualization", "Results","Info"])

            with tabs[0]:
                st.header("PCA Visualization")
               
                pca_visualization(data, target_variable)

            with tabs[1]:
                st.header("t-SNE Visualization")
                perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, key="tsne_perplexity_slider")
                learning_rate = st.slider("Learning Rate", min_value=10, max_value=1000, value=200, key="tsne_learning_rate_slider")
                t_sne_visualization(data, target_variable, perplexity, learning_rate)

            with tabs[2]:
                st.header("Feature Visualization")
                feature_visualization(data)

            # Results Tab
            with tabs[3]:
                
                features = data.columns.drop(target_variable)
                numeric_features = data[features].select_dtypes(include=np.number)
                categorical_features = data[features].select_dtypes(exclude=np.number)

                numeric_transformer = StandardScaler()
                categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features.columns),
                        ('cat', categorical_transformer, categorical_features.columns)])

                X = preprocessor.fit_transform(data[features])
                y = data[target_variable]

                # Train-test split
                X_train, X_test, y_train, y_test = split_data(X, y)

                # --- Individual Classifier Selection and Evaluation ---
                st.subheader("Classification")
                st.write("Όρισε τις παραμέτρους ταξινόμισης:")
                classifier_name = st.selectbox("Διάλεξε Classifier", ["K-Nearest Neighbors (KNN)", "Decision Tree"], key="classifier_selectbox")

                # Initialize classification_results here
                classification_results = None


                if classifier_name == "K-Nearest Neighbors (KNN)":
                    n_neighbors = st.number_input("Number of Neighbors (K)", min_value=1, value=5, key="knn_neighbors_input")
                    weights = st.selectbox("Weights", ['uniform', 'distance'], key="knn_weights_selectbox")
                    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                else:
                    max_depth = st.number_input("Maximum Depth", min_value=1, value=5, key="dt_max_depth_input")
                    criterion = st.selectbox("Criterion", ['gini', 'entropy'], key="dt_criterion_selectbox")
                    classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

                # Train and evaluate classifier
               
                classification_results = train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

                # Display classification results
                st.subheader("Αποτελέσματα Classification ")
                display_classification_results(classification_results)

                # Clustering
                st.subheader("Clustering")
                st.write("Όρισε τις παραμέτρους Clustering:")
                algorithm = st.selectbox("Διάλεξε Clustering Algorithm", ["KMeans", "Agglomerative Clustering"], key="clustering_algorithm_selectbox")

                # Algorithm-specific parameters
                if algorithm == "KMeans":
                    n_clusters = st.number_input("Number of Clusters (K) for K-Means", min_value=2, value=3, key="n_clusters_kmeans_input")
                    init = st.selectbox("Initialization Method", ['k-means++', 'random'], key="kmeans_init_selectbox")
                    clustering_params = {"n_clusters": n_clusters, "init": init, "random_state": 42}
                else:
                    n_clusters = st.number_input("Number of Clusters (K) for Agglomerative", min_value=2, value=3, key="n_clusters_agg_input")
                    linkage = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'], key="agg_linkage_selectbox")
                    clustering_params = {"n_clusters": n_clusters, "linkage": linkage}

                # Perform clustering
               
                clustering_model, clusters = perform_clustering(algorithm, X, **clustering_params) 
                clustering_results = evaluate_clustering(X, clusters, y)

                # Display clustering results
                st.subheader("Αποτελέσματα Clustering ")
                display_clustering_results(clustering_results)

                # Comparison section
                st.header("Σύγκριση Αποτελεσμάτων")
                st.write("Σύγκριση Επίδοσης Αλγορίθμων")
                
                # --- Classification Comparison ---
                
                compare_classifiers(X_train, y_train, X_test, y_test)

                # --- Clustering Comparison ---
                compare_clustering_algorithms(X, y)
                # Compare classification and clustering
                if classification_results["accuracy"] > clustering_results["silhouette_score"]:
                    st.write("Η μέθοδος Classification  απέδωσε καλύτερ accuracy ", classification_results["accuracy"])
                else:
                    st.write("Η μέθοδος Clustering απέδωσε καλύτερα με silhouette score of ", clustering_results["silhouette_score"])
               
        
        with tabs[4]:  # Index 4 for the "Info" tab
                   st.header("Σχετικά με αυτή την Εφαρμογή")
                   st.write(
                      """
        Αυτή η εφαρμογή έχει σχεδιαστεί για την οπτικοποίηση και αξιολόγηση μοντέλων μηχανικής μάθησης αλγορίθμων ταξινόμησης και ομαδοποίησης. 
        Παρέχει εργαλεία για την εξερεύνηση δεδομένων (PCA και t-SNE), την οπτικοποίηση χαρακτηριστικών και τη σύγκριση μοντέλων.
        """
    )

    st.subheader("Πώς Λειτουργεί")
    st.write(
        """
        Η εφαρμογή αποτελείται από 4 καρτέλες (tabs), καθεμία από τις οποίες παρέχει διαφορετική λειτουργικότητα.
       ### Sidebar: Φόρτωση Δεδομένων
        - **Περιγραφή:** Σε αυτή την καρτέλα μπορείτε να ανεβάσετε το σύνολο δεδομένων σας σε μορφή CSV.
        - **Πώς Λειτουργεί:** Απλά επιλέξτε το αρχείο σας και η εφαρμογή θα φορτώσει και θα εμφανίσει τα δεδομένα στον πίνακα. Βεβαιωθείτε ότι το σύνολο δεδομένων σας είναι καθαρό και σωστά μορφοποιημένο πριν το ανεβάσετε.

        ### Tab 2: PCA Ανάλυση
        - **Περιγραφή:** Χρησιμοποιήστε την ανάλυση PCA για να δείτε πώς φαίνονται τα δεδομένα σας σε δύο κύριες συνιστώσες.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Στηλών:** Μπορείτε να επιλέξετε ποιες στήλες από το σύνολο δεδομένων σας θα χρησιμοποιηθούν για την ανάλυση PCA.
          - **Οπτικοποίηση:** Το διάγραμμα που προκύπτει θα σας δείξει τα δεδομένα σας στις δύο κύριες συνιστώσες.
        
        ### Tab 3: t-SNE Ανάλυση
        - **Περιγραφή:** Χρησιμοποιήστε την ανάλυση t-SNE για μια μη γραμμική προβολή των δεδομένων σας.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Στηλών:** Επιλέξτε ποιες στήλες θα χρησιμοποιηθούν για την ανάλυση t-SNE.
          - **Ρυθμίσεις Παραμέτρων:** Μπορείτε να προσαρμόσετε τις παραμέτρους όπως:
            - **Perplexity:** Επιλέξτε τιμή στο εύρος από 5 έως 50.
            - **Learning Rate:** Επιλέξτε τιμή στο εύρος από 10 έως 1000.
          - **Οπτικοποίηση:** Το διάγραμμα που προκύπτει θα σας δείξει τα δεδομένα σας σε μια μη γραμμική προβολή.

        ### Tab 4: Οπτικοποίηση Χαρακτηριστικών
        - **Περιγραφή:** Οπτικοποιήστε τις κατανομές και τις σχέσεις των επιμέρους χαρακτηριστικών με τη μεταβλητή στόχου.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Χαρακτηριστικού:** Επιλέξτε το χαρακτηριστικό που θέλετε να οπτικοποιήσετε.
          - **Οπτικοποίηση:** Το διάγραμμα θα σας δείξει την κατανομή ή τη σχέση του επιλεγμένου χαρακτηριστικού με τη μεταβλητή στόχου.

        ### Tab 5: Αξιολόγηση Ταξινόμησης
        - **Περιγραφή:** Εκπαιδεύστε και αξιολογήστε μοντέλα ταξινόμησης όπως KNN και Δέντρο Αποφάσεων.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Αλγορίθμου:** Επιλέξτε τον αλγόριθμο ταξινόμησης που θέλετε να χρησιμοποιήσετε.
          - **Ρυθμίσεις Παραμέτρων:** Μπορείτε να προσαρμόσετε τις παραμέτρους του αλγορίθμου.
          - **Εκπαίδευση και Αξιολόγηση:** Εκπαιδεύστε το μοντέλο και δείτε τα αποτελέσματα όπως η ακρίβεια, το precision και το recall.

        ### Tab 6: Αξιολόγηση Ομαδοποίησης
        - **Περιγραφή:** Εκτελέστε ομαδοποίηση με KMeans και Ιεραρχική Ομαδοποίηση και αξιολογήστε τα αποτελέσματα.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Αλγορίθμου:** Επιλέξτε τον αλγόριθμο ομαδοποίησης που θέλετε να χρησιμοποιήσετε.
          - **Ρυθμίσεις Παραμέτρων:** Μπορείτε να προσαρμόσετε τις παραμέτρους του αλγορίθμου όπως ο αριθμός των clusters.
          - **Εκπαίδευση και Αξιολόγηση:** Εκτελέστε την ομαδοποίηση και δείτε τα αποτελέσματα όπως το silhouette score και το inertia.

        ### Tab 7: Σύγκριση Μοντέλων
        - **Περιγραφή:** Συγκρίνετε διάφορους αλγορίθμους ταξινόμησης και ομαδοποίησης για να βρείτε τον καλύτερο για τα δεδομένα σας.
        - **Πώς Λειτουργεί:**
          - **Επιλογή Μοντέλων:** Επιλέξτε τα μοντέλα που θέλετε να συγκρίνετε.
          - **Σύγκριση:** Δείτε συγκριτικά αποτελέσματα όπως η ακρίβεια και άλλες μετρήσεις απόδοσης για κάθε μοντέλο.
        """
    )

    st.subheader("Χαρακτηριστικά")
    st.write(
        """
        - **Φόρτωση και Εμφάνιση Δεδομένων:** Ανεβάστε αρχεία CSV και δείτε το σύνολο δεδομένων.
        - **Οπτικοποίηση PCA:** Δείτε πώς φαίνονται τα δεδομένα σας σε δύο κύριες συνιστώσες.
        - **Οπτικοποίηση t-SNE:** Λάβετε μια μη γραμμική προβολή των δεδομένων σας.
        - **Οπτικοποίηση Χαρακτηριστικών:** Οπτικοποιήστε τις κατανομές και τις σχέσεις των επιμέρους χαρακτηριστικών.
        - **Ταξινόμηση:** Εκπαιδεύστε και αξιολογήστε μοντέλα ταξινόμησης όπως KNN και Δέντρο Αποφάσεων.
        - **Ομαδοποίηση:** Εκτελέστε ομαδοποίηση με KMeans και Ιεραρχική Ομαδοποίηση και αξιολογήστε τα αποτελέσματα.
        - **Σύγκριση Μοντέλων:** Συγκρίνετε διάφορους αλγορίθμους για να βρείτε τον καλύτερο απόδοσης μοντέλο.
        """
    )

    st.subheader("Συμβουλές για τη Χρήση της Εφαρμογής")
    st.write(
        """
        - Βεβαιωθείτε ότι το σύνολο δεδομένων σας είναι καθαρό και σωστά μορφοποιημένο πριν το ανεβάσετε.
        - Επιλέξτε την κατάλληλη μεταβλητή στόχου για την ανάλυσή σας.
        - Χρησιμοποιήστε τους ρυθμιστές και τα κουμπιά επιλογής για να προσαρμόσετε τις παραμέτρους του μοντέλου για καλύτερα αποτελέσματα.
        - Συγκρίνετε τα αποτελέσματα διαφορετικών μοντέλων για να λάβετε μια τεκμηριωμένη απόφαση.
        """
    )

    st.subheader("Ομάδα Ανάπτυξης")
    st.write(
        """
        Αυτή η εφαρμογή αναπτύχθηκε από:

        * **Μέλος Ομάδας 1:** 
            - Υλοποίησε τη φόρτωση και προεπεξεργασία δεδομένων.
            - Ανέπτυξε τα εργαλεία οπτικοποίησης PCA και t-SNE.
        * **Μέλος Ομάδας 2:**
        """
    )
