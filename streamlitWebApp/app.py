import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import umap
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.markdown("""
    <style>
    /* Γενικό στυλ για το φόντο */
  body {
    background-color: #e6e2d3; /* Ανοιχτό κρεμ χρώμα */
    color: black; /* Μαύρα γράμματα */
}
.stApp {
     background: linear-gradient(135deg, #333 70%, white 100%) !important;
}
header {
        background-color: 135deg;
    }
/* Στυλ για τα Tabs */
.stTabs [role="tablist"] {
    display: flex;
    justify-content: flex-start;
    border-bottom: 1px solid black;
}

/* Στυλ για το κάθε Tab */
.stTabs [role="tab"] {
    padding: 12px 24px;
    border: 1px solid black; /* Μαύρο περίγραμμα για τα Tabs */
     background: linear-gradient(135deg, #333 100%, white 50%) !important;
 /* Λευκό φόντο για τα tabs */
    margin-right: 8px;
    style:bold;
    color: black;
    font-weight: bold;
}

/* Στυλ για ενεργό Tab */
.stTabs [role="tab"][aria-selected="true"] {
    background-color: white;
    border-bottom: 3px solid black; /* Έντονο για το ενεργό tab */
}

/* Στυλ για τα headers */
h1, h2, h3, h4 {
    color: black;
}

/* Αποφυγή αλλαγής διάταξης των κουμπιών και της φόρτωσης αρχείων */
.stFileUploader, .stButton {
    color: black;
}
    </style>
""", unsafe_allow_html=True)

# Tab structure
tab1, tab2, tab3, tab4, tab5= st.tabs(["Data Upload", "Visualization", "Feature Selection", "Classification", "Info"])


# Tab 1: Data upload
with tab1:
    st.title("Εφαρμογή Εξόρυξης και Ανάλυσης Δεδομένων")
    
    # Φόρτωση αρχείου δεδομένων
    uploaded_file = st.file_uploader("Φόρτωση αρχείου (CSV, Excel, TSV)", type=['csv', 'xlsx', 'tsv'])
    
    if uploaded_file:
        # Φόρτωση του DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, delimiter='\t')
             # Ζητάμε από τον χρήστη να επιλέξει τη στήλη στόχο
        target_column = st.selectbox("Επιλέξτε τη στήλη που περιέχει τη μεταβλητή στόχο:", df.columns)
        # Εμφάνιση αρχικού dataset
        st.write("Αρχικό Dataset:")
        st.dataframe(df)

   
        
        # Μετακίνηση της στήλης στόχου στην τελευταία θέση
        if target_column and target_column != df.columns[-1]:
            features = df.drop(columns=[target_column])
            df_reordered = pd.concat([features, df[target_column]], axis=1)
        else:
            df_reordered = df

        # Εμφάνιση του αναδιοργανωμένου dataset
        st.write("Dataset με τη στήλη στόχο στην τελευταία θέση:")
        st.dataframe(df_reordered)

        X = df_reordered.iloc[:, :-1]  
        y = df_reordered.iloc[:, -1]  

        # Κωδικοποίηση κατηγορικών μεταβλητών
        if X.select_dtypes(include=['object']).shape[1] > 0:
            X_encoded = pd.get_dummies(X)  # Κωδικοποίηση κατηγορικών μεταβλητών
        else:
            X_encoded = X
        
        
    else:
        st.write("Παρακαλώ ανεβάστε ένα αρχείο.")
# Tab 2: Visualization
with tab2:
    if uploaded_file:
        st.title("Οπτικοποιήσεις")

        # PCA για 2D
        st.subheader("PCA (2D)")
        pca_2d = PCA(n_components=2)
        pca_result_2d = pca_2d.fit_transform(X_encoded)  

        fig_pca_2d = px.scatter(x=pca_result_2d[:, 0], y=pca_result_2d[:, 1], color=y,
                                title="2D PCA",
                                labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'})
        st.plotly_chart(fig_pca_2d)
        
        # Επεξήγηση PCA (2D)
        st.write("""
        **PCA (2D):** Το Principal Component Analysis (PCA) είναι μια τεχνική μείωσης διάστασης που μετασχηματίζει τα δεδομένα σε έναν νέο χώρο μικρότερων διαστάσεων, 
        διατηρώντας όσο το δυνατόν περισσότερη πληροφορία. Εδώ τα δεδομένα απεικονίζονται σε δύο διαστάσεις, όπου οι δύο κύριες συνιστώσες αιχμαλωτίζουν το μεγαλύτερο ποσοστό 
        της διασποράς των αρχικών δεδομένων.
        """)

        # PCA για 3D
        st.subheader("PCA (3D)")
        pca_3d = PCA(n_components=3)
        pca_result_3d = pca_3d.fit_transform(X_encoded)

        fig_pca_3d = px.scatter_3d(x=pca_result_3d[:, 0], y=pca_result_3d[:, 1], z=pca_result_3d[:, 2], 
                                   color=y,
                                   title="3D PCA",
                                   labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'z': 'Principal Component 3'})
        st.plotly_chart(fig_pca_3d)

        # Επεξήγηση PCA (3D)
        st.write("""
        **PCA (3D):** Το 3D PCA λειτουργεί παρόμοια με το 2D, αλλά η απεικόνιση γίνεται σε τρεις διαστάσεις. Αυτό βοηθά στην καλύτερη απεικόνιση των δεδομένων, 
        όταν τα δύο πρώτα κύρια στοιχεία δεν επαρκούν για την πλήρη περιγραφή της διασποράς των δεδομένων.
        """)

        # UMAP για 2D
        st.subheader("UMAP (2D)")
        reducer_2d = umap.UMAP(n_components=2)
        umap_result_2d = reducer_2d.fit_transform(X_encoded)

        fig_umap_2d = px.scatter(x=umap_result_2d[:, 0], y=umap_result_2d[:, 1], color=y,
                                 title="2D UMAP",
                                 labels={'x': 'UMAP 1', 'y': 'UMAP 2'})
        st.plotly_chart(fig_umap_2d)

        # Επεξήγηση UMAP (2D)
        st.write("""
        **UMAP (2D):** Το UMAP (Uniform Manifold Approximation and Projection) είναι μια άλλη τεχνική μείωσης διάστασης που προσπαθεί να διατηρήσει τόσο την τοπική 
        όσο και την παγκόσμια δομή των δεδομένων. Εδώ τα δεδομένα προβάλλονται σε δύο διαστάσεις, διατηρώντας την εγγύτητα μεταξύ των σημείων που υπήρχε στον αρχικό χώρο.
        """)

        # UMAP για 3D
        st.subheader("UMAP (3D)")
        reducer_3d = umap.UMAP(n_components=3)
        umap_result_3d = reducer_3d.fit_transform(X_encoded)

        fig_umap_3d = px.scatter_3d(x=umap_result_3d[:, 0], y=umap_result_3d[:, 1], z=umap_result_3d[:, 2], 
                                    color=y,
                                    title="3D UMAP",
                                    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'})
        st.plotly_chart(fig_umap_3d)

        # Επεξήγηση UMAP (3D)
        st.write("""
        **UMAP (3D):** Το UMAP σε τρεις διαστάσεις επιτρέπει την καλύτερη απεικόνιση των δεδομένων με την προσθήκη μιας τρίτης διάστασης, 
        κάτι που μπορεί να βοηθήσει στην καλύτερη κατανόηση των σχέσεων μεταξύ των σημείων δεδομένων, διατηρώντας τόσο τη δομή όσο και τις συσχετίσεις τους.
        """)

        # Exploratory Data Analysis (EDA) - Boxplot και Heatmap
        st.title("Exploratory Data Analysis (EDA)")

        # Boxplot
        st.subheader("Boxplot")
        feature = st.selectbox("Επιλέξτε χαρακτηριστικό για Boxplot", X_encoded.columns)
        fig_boxplot = plt.figure(figsize=(10, 6))
        sns.boxplot(x=X_encoded[feature])
        st.pyplot(fig_boxplot)

        # Επεξήγηση Boxplot
        st.write("""
        **Boxplot:** Το Boxplot (ή διάγραμμα κουτιού) απεικονίζει την κατανομή ενός χαρακτηριστικού. Παρουσιάζει τον διάμεσο, το πρώτο και τρίτο τεταρτημόριο, 
        καθώς και τα ακραία σημεία (outliers), προσφέροντας μια γρήγορη εικόνα της κατανομής των δεδομένων.
        """)

        # Heatmap
        st.subheader("Heatmap")
        corr = X_encoded.corr()  
        fig_heatmap = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(fig_heatmap)

        # Επεξήγηση Heatmap
        st.write("""
        **Heatmap:** Το Heatmap (διάγραμμα θερμοκρασίας) απεικονίζει τη συσχέτιση μεταξύ των χαρακτηριστικών. Οι υψηλές συσχετίσεις (θετικές ή αρνητικές) 
        αναδεικνύονται με έντονα χρώματα, επιτρέποντας την ανίχνευση σχέσεων μεταξύ των χαρακτηριστικών.
        """)

# Tab Feature Selection
with tab3:
    st.title("Feature Selection")
    
    # Αρχικό dataset από προηγούμενα tabs
    if 'df' in locals():  # Αν έχει ήδη φορτωθεί dataset στο προηγούμενο tab
        st.write("Αρχικό Dataset:")
        st.dataframe(df)
        
        # Επιλογή του αριθμού των χαρακτηριστικών από τον χρήστη
        num_features = st.slider("Επιλέξτε τον αριθμό των χαρακτηριστικών:", min_value=1, max_value=df.shape[1]-1, value=5)
        
        # Διαδικασία Feature Selection με χρήση SelectKBest
        X = df_reordered.iloc[:, :-1]  # Χωρίς την τελευταία στήλη (ετικέτες)
        y = df_reordered.iloc[:, -1]  # Τελευταία στήλη (ετικέτες)

        # Κωδικοποίηση κατηγορικών μεταβλητών
        if X.select_dtypes(include=['object']).shape[1] > 0:
            X_encoded = pd.get_dummies(X)  # Κωδικοποίηση κατηγορικών μεταβλητών
        else:
            X_encoded = X
        
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X_new = selector.fit_transform(X_encoded, y)
        
        st.write(f"Top {num_features} Χαρακτηριστικά:")
        top_features = X_encoded.columns[selector.get_support()]
        st.write(top_features)
        
        # Dataset με τα επιλεγμένα χαρακτηριστικά
        st.write("Dataset με τα επιλεγμένα χαρακτηριστικά:")
        df_selected = pd.DataFrame(X_new, columns=top_features)
        df_selected['Label'] = y.values  # Προσθήκη της ετικέτας
        st.dataframe(df_selected)
# Tab 4: Classification
with tab4:
    st.title("Κατηγοριοποιήση")

    if 'df' in locals() and target_column:
        # Επιλογή παραμέτρου k για KNN
        k = st.slider("Επιλέξτε το 'k' για τον αλγόριθμο K-Nearest Neighbors:", min_value=1, max_value=15, value=5)
        
        # Επιλογή παραμέτρου n_estimators για Random Forest
        n_estimators = st.slider("Επιλέξτε τον αριθμό δέντρων για τον Random Forest:", min_value=10, max_value=200, value=100)
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
        X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(df_selected.iloc[:, :-1], 
                                                                                                df_selected['Label'], 
                                                                                                test_size=0.3, random_state=42)
        
        # Αλγόριθμος 1: K-Nearest Neighbors (KNN)
        knn_before = KNeighborsClassifier(n_neighbors=k)
        knn_after = KNeighborsClassifier(n_neighbors=k)
        
        knn_before.fit(X_train, y_train)
        knn_after.fit(X_train_selected, y_train_selected)

        y_pred_before = knn_before.predict(X_test)
        y_pred_after = knn_after.predict(X_test_selected)
        
        # Αλγόριθμος 2: Random Forest
        rf_before = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_after = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        rf_before.fit(X_train, y_train)
        rf_after.fit(X_train_selected, y_train_selected)

        y_pred_rf_before = rf_before.predict(X_test)
        y_pred_rf_after = rf_after.predict(X_test_selected)

        # Υπολογισμός μετρικών απόδοσης
        def compute_metrics(y_true, y_pred, y_prob):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_prob[:, 1])
            return accuracy, f1, roc_auc

        # Μετρικές πριν
        y_prob_before_knn = knn_before.predict_proba(X_test)
        accuracy_before_knn, f1_before_knn, roc_auc_before_knn = compute_metrics(y_test, y_pred_before, y_prob_before_knn)

        y_prob_before_rf = rf_before.predict_proba(X_test)
        accuracy_before_rf, f1_before_rf, roc_auc_before_rf = compute_metrics(y_test, y_pred_rf_before, y_prob_before_rf)

        # Μετρικές μετά
        y_prob_after_knn = knn_after.predict_proba(X_test_selected)
        accuracy_after_knn, f1_after_knn, roc_auc_after_knn = compute_metrics(y_test_selected, y_pred_after, y_prob_after_knn)

        y_prob_after_rf = rf_after.predict_proba(X_test_selected)
        accuracy_after_rf, f1_after_rf, roc_auc_after_rf = compute_metrics(y_test_selected, y_pred_rf_after, y_prob_after_rf)

        # Αποτελέσματα K-Nearest Neighbors
        st.subheader("Αποτελέσματα K-Nearest Neighbors")
        st.write(f"**Accuracy (πριν την επιλογή χαρακτηριστικών):** {accuracy_before_knn:.2f}")
        st.write(f"**Accuracy (μετά την επιλογή χαρακτηριστικών):** {accuracy_after_knn:.2f}")
        st.write(f"**F1-Score (πριν):** {f1_before_knn:.2f}")
        st.write(f"**F1-Score (μετά):** {f1_after_knn:.2f}")
        st.write(f"**ROC-AUC (πριν):** {roc_auc_before_knn:.2f}")
        st.write(f"**ROC-AUC (μετά):** {roc_auc_after_knn:.2f}")

        # Συγκριτική ανάλυση KNN
        st.write(f"**Διαφορά Ακρίβειας (KNN):** {accuracy_after_knn - accuracy_before_knn:.2f}")
        st.write(f"**Διαφορά F1-Score (KNN):** {f1_after_knn - f1_before_knn:.2f}")
        st.write(f"**Διαφορά ROC-AUC (KNN):** {roc_auc_after_knn - roc_auc_before_knn:.2f}")

        # Αποτελέσματα Random Forest
        st.subheader("Αποτελέσματα Random Forest")
        st.write(f"**Accuracy (πριν):** {accuracy_before_rf:.2f}")
        st.write(f"**Accuracy (μετά):** {accuracy_after_rf:.2f}")
        st.write(f"**F1-Score (πριν):** {f1_before_rf:.2f}")
        st.write(f"**F1-Score (μετά):** {f1_after_rf:.2f}")
        st.write(f"**ROC-AUC (πριν):** {roc_auc_before_rf:.2f}")
        st.write(f"**ROC-AUC (μετά):** {roc_auc_after_rf:.2f}")

        # Συγκριτική ανάλυση Random Forest
        st.write(f"**Διαφορά Ακρίβειας (Random Forest):** {accuracy_after_rf - accuracy_before_rf:.2f}")
        st.write(f"**Διαφορά F1-Score (Random Forest):** {f1_after_rf - f1_before_rf:.2f}")
        st.write(f"**Διαφορά ROC-AUC (Random Forest):** {roc_auc_after_rf - roc_auc_before_rf:.2f}")

        # Νέο section για σύγκριση αλγορίθμων
        st.title("Σύγκριση Απόδοσης Αλγορίθμων")

        # Δημιουργία πίνακα σύγκρισης
        comparison_df = pd.DataFrame({
            "Αλγόριθμος": ["KNN", "Random Forest"],
            "Accuracy (Πριν)": [accuracy_before_knn, accuracy_before_rf],
            "F1-Score (Πριν)": [f1_before_knn, f1_before_rf],
            "ROC-AUC (Πριν)": [roc_auc_before_knn, roc_auc_before_rf],
            "Accuracy (Μετά)": [accuracy_after_knn, accuracy_after_rf],
            "F1-Score (Μετά)": [f1_after_knn, f1_after_rf],
            "ROC-AUC (Μετά)": [roc_auc_after_knn, roc_auc_after_rf]
        })

        st.write(comparison_df)

        # Υπολογισμός του καλύτερου αλγορίθμου ανά μέτρηση
        best_accuracy_algo = comparison_df.loc[comparison_df[['Accuracy (Πριν)', 'Accuracy (Μετά)']].idxmax().max()]["Αλγόριθμος"]
        best_f1_algo = comparison_df.loc[comparison_df[['F1-Score (Πριν)', 'F1-Score (Μετά)']].idxmax().max()]["Αλγόριθμος"]
        best_roc_auc_algo = comparison_df.loc[comparison_df[['ROC-AUC (Πριν)', 'ROC-AUC (Μετά)']].idxmax().max()]["Αλγόριθμος"]
        st.write(f"**Καλύτερος αλγόριθμος με βάση την Accuracy:** {best_accuracy_algo}")
        st.write(f"**Καλύτερος αλγόριθμος με βάση το F1-Score:** {best_f1_algo}")
        st.write(f"**Καλύτερος αλγόριθμος με βάση το ROC-AUC:** {best_roc_auc_algo}")
# Tab Info: Πληροφορίες για την Εφαρμογή
with tab5:
    st.title("Πληροφορίες Εφαρμογής")

    st.write("""
    ### Εφαρμογή Εξόρυξης και Ανάλυσης Δεδομένων
    Αυτή η εφαρμογή έχει σχεδιαστεί για την επεξεργασία, οπτικοποίηση, επιλογή χαρακτηριστικών και κατηγοριοποίηση δεδομένων. Παρέχει ευέλικτα εργαλεία για την ανάλυση και την εξόρυξη γνώσης από δεδομένα.

    ### Τρόπος Λειτουργίας:
    - **Data Upload**: Ο χρήστης μπορεί να ανεβάσει δεδομένα σε μορφή CSV, Excel, ή TSV για ανάλυση.ΠΡΟΣΟΧΗ.Η ΕΦΑΡΜΟΓΗ ΕΠΙΤΡΕΠΕΙ Η ΜΕΤΑΒΛΗΤΗ ΣΤΟΧΟΣ ΝΑ ΕΙΝΑΙ ΤΥΠΟΥ ΚΑΤΗΓΟΡHΜΑΤΙΚΗ.
    - **Visualization**: Προσφέρεται οπτικοποίηση των δεδομένων μέσω διαγραμμάτων PCA, UMAP και άλλων εργαλείων.
    - **Feature Selection**: Παρέχει επιλογή χαρακτηριστικών με χρήση μεθόδων όπως το SelectKBest.Ο χρήστης μπορεί να ορίσει πόσα features θέλει να κρατήσει μέσω της μπάρας
    - **Classification**: Υποστηρίζει την εφαρμογή αλγορίθμων κατηγοριοποίησης όπως KNN και Random Forest.Ο χρήστης μπορεί να ορίσει μέσω της μπάρας των αριθμό k αλλα και ον αριθμό δέντρων για τον Random Forest.

    ### Ομάδα Ανάπτυξης:
    - **Μπάρλας Ιωάννης Π2019009**: Ανάπτυξη εξ΄ολοκλήρου της εφαρμογής.
    
    """)
