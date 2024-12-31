import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import io
import tempfile
import os
import math  # Importation du module math pour floor()

# ------------------------------
# 1. Configuration de la Page
# ------------------------------

st.set_page_config(
    page_title="🔍 Customer Comments Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# 2. Importation des Données et Prétraitement
# ------------------------------

# Télécharger les stopwords de NLTK si non déjà téléchargés
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Nettoie le texte en minuscules, supprime les caractères non alphanumériques,
    les chiffres et les stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def calculate_weighted_score(row, alpha=0.04, beta=0.01):
    """
    Calcule un score pondéré basé sur le score initial, les likes et les dislikes.
    Utilise une formule unifiée pour assurer la consistance.
    """
    weighted_score = row['Score'] * (1 + alpha * row['likes'] - beta * row['dislikes'])
    # Borne du score pondéré entre 1 et 5
    weighted_score = max(1, min(weighted_score, 5))
    # Arrondissement par troncature
    score_final = math.floor(weighted_score)
    return score_final

def clean_comment_for_pdf(text):
    """
    Supprime les caractères non-ASCII du texte pour la compatibilité PDF.
    """
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# ------------------------------
# 3. Fonction pour Générer le PDF
# ------------------------------

def generate_pdf(visualizations, score_global, recommendation, path="comments_report.pdf"):
    """
    Génère un rapport PDF contenant les visualisations et leurs commentaires.
    Inclut une recommandation basée sur le score global.
    Supprime automatiquement les fichiers temporaires après la génération.
    """
    if not visualizations:
        raise ValueError("La liste des visualisations est vide. Aucun PDF ne peut être généré.")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Ajouter les variantes de la police DejaVu
    font_path_regular = os.path.join('fonts', 'DejaVuSans.ttf')
    font_path_bold = os.path.join('fonts', 'DejaVuSans-Bold.ttf')
    
    # Vérifier la présence des fichiers de police
    if not os.path.exists(font_path_regular):
        raise FileNotFoundError(f"Font file {font_path_regular} not found.")
    if not os.path.exists(font_path_bold):
        raise FileNotFoundError(f"Font file {font_path_bold} not found.")
    
    # Ajouter la police régulière et en gras
    pdf.add_font("DejaVu", '', font_path_regular, uni=True)
    pdf.add_font("DejaVu", 'B', font_path_bold, uni=True)
    
    # Définir la police par défaut
    pdf.set_font("DejaVu", size=12)
    
    # Titre
    pdf.set_font("DejaVu", 'B', 16)
    pdf.cell(0, 10, txt="Customer Comments Analysis Report", ln=True, align='C')
    
    pdf.ln(10)  # Saut de ligne
    
    # Score global ajusté
    pdf.set_font("DejaVu", 'B', 12)
    pdf.cell(0, 10, txt=f"Adjusted Global Score: {score_global}", ln=True)
    
    # Recommandation basée sur le score global
    pdf.ln(5)  # Petit saut de ligne
    pdf.set_font("DejaVu", 'B', 14)
    pdf.multi_cell(0, 10, txt="📋 Recommandation:")
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, txt=recommendation)
    
    pdf.ln(10)  # Saut de ligne
    
    # ------------------------------
    # 3.2. Ajouter les Visualisations avec Commentaires
    # ------------------------------
    
    for idx, viz in enumerate(visualizations):
        # Vérification de la structure de chaque visualisation
        if not isinstance(viz, (list, tuple)) or len(viz) != 2:
            st.warning(f"⚠️ Élément mal formé dans 'visualizations' à l'index {idx} : {viz}")
            continue
        img_path, comment = viz
        # Convertir les backslashes en slashes pour éviter les problèmes avec FPDF
        img_path = img_path.replace('\\', '/')
        if os.path.exists(img_path):
            try:
                # Ajouter le commentaire
                pdf.set_font("DejaVu", 'B', 14)
                pdf.multi_cell(0, 10, txt=comment)
                
                # Ajouter l'image
                pdf.image(img_path, w=190)  # Ajuster la largeur selon vos besoins
                pdf.ln(10)  # Saut de ligne après chaque image
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement de la visualisation {idx + 1} : {e}")
                continue
        else:
            st.warning(f"⚠️ Fichier image manquant pour la visualisation {idx + 1} : {img_path}")
    
    # Sauvegarder le PDF en mémoire
    try:
        pdf_content = pdf.output(dest='S').encode('latin1')  # Encode en bytes
        pdf_buffer = io.BytesIO(pdf_content)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la sauvegarde du PDF : {e}")
    
    # Retourner le buffer pour téléchargement
    return pdf_buffer

# ------------------------------
# 4. Initialisation de la Session
# ------------------------------

# Initialiser session_state pour les commentaires manuels
if 'manual_comments' not in st.session_state:
    st.session_state.manual_comments = []
if 'manual_likes' not in st.session_state:
    st.session_state.manual_likes = []
if 'manual_dislikes' not in st.session_state:
    st.session_state.manual_dislikes = []

# Initialiser session_state pour les commentaires importés
if 'imported_comments' not in st.session_state:
    st.session_state.imported_comments = []
if 'imported_likes' not in st.session_state:
    st.session_state.imported_likes = []
if 'imported_dislikes' not in st.session_state:
    st.session_state.imported_dislikes = []

# Initialiser session_state pour l'engagement de la publication
if 'publication_engagement_submitted' not in st.session_state:
    st.session_state.publication_engagement_submitted = False
if 'total_likes_publication' not in st.session_state:
    st.session_state.total_likes_publication = 0
if 'total_dislikes_publication' not in st.session_state:
    st.session_state.total_dislikes_publication = 0

# Initialiser session_state pour les résultats analysés
if 'analyzed_df' not in st.session_state:
    st.session_state.analyzed_df = None
if 'adjusted_global_score' not in st.session_state:
    st.session_state.adjusted_global_score = None
if 'global_recommendation' not in st.session_state:
    st.session_state.global_recommendation = ""

# ------------------------------
# 5. Interface Utilisateur avec Onglets
# ------------------------------

# CSS personnalisé pour le style
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        color: #4CAF50;
    }
    .medium-font {
        font-size:18px !important;
        color: #2196F3;
    }
    .small-font {
        font-size:14px !important;
        color: #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.markdown('<h1 class="big-font">📈 Customer Comments Analysis and Management</h1>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">This application allows you to add and import customer comments, perform in-depth analysis, and generate interactive visualizations along with PDF reports.</p>', unsafe_allow_html=True)

# Créer les onglets
tabs = st.tabs(["📥 Chargement des Commentaires", "📊 Résultats", "🎨 Tableau de Bord"])

# ------------------------------
# 5.1. Onglet "Chargement des Commentaires"
# ------------------------------
with tabs[0]:
    st.subheader("📥 Chargement des Commentaires")
    
    # Formulaire pour l'engagement de la publication
    with st.form("publication_engagement_form"):
        st.write("### Saisissez les likes et dislikes sur la publication")
        total_likes_publication = st.number_input("👍 Total Likes sur la Publication", min_value=0, step=1, value=0, key="input_total_likes_publication")
        total_dislikes_publication = st.number_input("👎 Total Dislikes sur la Publication", min_value=0, step=1, value=0, key="input_total_dislikes_publication")
        
        submitted_publication = st.form_submit_button("✅ Soumettre l'Engagement de la Publication")
        
        if submitted_publication:
            # Stocker les valeurs dans session_state avec des clés distinctes
            st.session_state.total_likes_publication = int(total_likes_publication)
            st.session_state.total_dislikes_publication = int(total_dislikes_publication)
            st.session_state.publication_engagement_submitted = True
            st.success("✅ Données d'engagement de la publication soumises avec succès!")
    
    st.markdown("---")
    
    # Formulaire pour ajouter des commentaires manuellement
    with st.form("add_multiple_comments_form"):
        st.write("### Ajouter des Commentaires Manuellement")
        num_manual = st.number_input("Nombre de commentaires à ajouter", min_value=1, max_value=20, value=1, step=1)
        
        for i in range(int(num_manual)):
            st.markdown(f"**Commentaire {i+1}**")
            comment = st.text_area(f"Commentaire {i+1}", height=80, placeholder="Écrivez votre commentaire ici...", key=f"manual_comment_{i}")
            likes = st.number_input(f"👍 Likes pour le Commentaire {i+1}", min_value=0, step=1, value=0, key=f"manual_likes_{i}")
            dislikes = st.number_input(f"👎 Dislikes pour le Commentaire {i+1}", min_value=0, step=1, value=0, key=f"manual_dislikes_{i}")
            st.markdown("---")
        
        submitted_manual = st.form_submit_button("🚀 Ajouter les Commentaires Manuels")
        
        if submitted_manual:
            for i in range(int(num_manual)):
                comment = st.session_state.get(f"manual_comment_{i}", "").strip()
                likes = st.session_state.get(f"manual_likes_{i}", 0)
                dislikes = st.session_state.get(f"manual_dislikes_{i}", 0)
                
                if comment == "":
                    st.warning(f"⚠️ Commentaire {i+1} est vide et sera ignoré.")
                    continue
                if likes < 0:
                    st.warning(f"⚠️ Les likes pour le Commentaire {i+1} doivent être des entiers positifs. Cette entrée sera ignorée.")
                    continue
                if dislikes < 0:
                    st.warning(f"⚠️ Les dislikes pour le Commentaire {i+1} doivent être des entiers positifs. Cette entrée sera ignorée.")
                    continue
                
                st.session_state.manual_comments.append(comment)
                st.session_state.manual_likes.append(int(likes))
                st.session_state.manual_dislikes.append(int(dislikes))
            
            st.success("✅ Commentaires manuels ajoutés avec succès!")

    st.markdown("---")
    
    # Section pour importer des commentaires depuis un fichier
    st.subheader("📂 Importer des Commentaires depuis un Fichier")
    
    uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel contenant des commentaires", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_imported = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df_imported = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel.")
                df_imported = None
            
            if df_imported is not None:
                required_columns = {'Text', 'likes', 'dislikes'}
                if not required_columns.issubset(df_imported.columns):
                    st.error(f"❌ Le fichier doit contenir les colonnes suivantes : {required_columns}")
                else:
                    st.subheader("🔍 Aperçu des Données Importées")
                    st.dataframe(df_imported.head())
                    
                    # Bouton pour importer les données
                    if st.button("📥 Importer les Données"):
                        for index, row in df_imported.iterrows():
                            comment = str(row['Text']).strip()
                            likes = row['likes']
                            dislikes = row['dislikes']
                            
                            if comment == "":
                                st.warning(f"⚠️ Ligne {index + 2} : Commentaire vide et sera ignoré.")
                                continue
                            if likes < 0:
                                st.warning(f"⚠️ Ligne {index + 2} : Les likes doivent être des entiers positifs. Cette ligne sera ignorée.")
                                continue
                            if dislikes < 0:
                                st.warning(f"⚠️ Ligne {index + 2} : Les dislikes doivent être des entiers positifs. Cette ligne sera ignorée.")
                                continue
                            
                            st.session_state.imported_comments.append(comment)
                            st.session_state.imported_likes.append(int(likes))
                            st.session_state.imported_dislikes.append(int(dislikes))
                        
                        st.success("✅ Données importées avec succès!")
        except Exception as e:
            st.error(f"❌ Une erreur est survenue lors de l'importation du fichier : {e}")

# ------------------------------
# 5.2. Onglet "Résultats"
# ------------------------------
with tabs[1]:
    st.subheader("📊 Résultats")
    
    # Vérifier si les données de la publication ont été soumises
    if st.session_state.publication_engagement_submitted:
        # Vérifier si des commentaires ont été ajoutés ou importés
        if (st.session_state.manual_comments or st.session_state.imported_comments):
            # Combiner les données manuelles et importées
            data_manual = pd.DataFrame({
                'Text': st.session_state.manual_comments,
                'likes': st.session_state.manual_likes,
                'dislikes': st.session_state.manual_dislikes
            })
            
            data_imported = pd.DataFrame({
                'Text': st.session_state.imported_comments,
                'likes': st.session_state.imported_likes,
                'dislikes': st.session_state.imported_dislikes
            })
            
            data_combined = pd.concat([data_manual, data_imported], ignore_index=True)
            
            st.write(f"**Nombre total de commentaires :** {data_combined.shape[0]}")
            
            # Bouton pour lancer l'analyse
            if st.button("🔍 Lancer l'Analyse"):
                # Définir les coefficients pour les likes et dislikes de la publication
                alpha = 0.04  # Coefficient pondérant l'impact des likes
                beta = 0.01   # Coefficient pondérant l'impact des dislikes
                
                # Vérifier la présence des fichiers de police
                font_path = os.path.join('fonts', 'DejaVuSans.ttf')
                font_path_bold = os.path.join('fonts', 'DejaVuSans-Bold.ttf')
                if not os.path.exists(font_path):
                    st.error("❌ Le fichier de police DejaVuSans.ttf est manquant. Veuillez le télécharger et le placer dans le répertoire 'fonts/'.")
                    st.stop()
                if not os.path.exists(font_path_bold):
                    st.error("❌ Le fichier de police DejaVuSans-Bold.ttf est manquant. Veuillez le télécharger et le placer dans le répertoire 'fonts/'.")
                    st.stop()
                else:
                    st.success("✅ Les fichiers de police DejaVuSans.ttf et DejaVuSans-Bold.ttf sont prêts à être utilisés.")
    
                # Charger le modèle et le vectoriseur (assurez-vous que les chemins sont corrects)
                try:
                    model = load_model('model_reviews.keras')  # Assurez-vous que le chemin est correct
                    with open('tfidf_vectorizer.pkl', 'rb') as f:
                        vectorizer = pickle.load(f)
                except FileNotFoundError as e:
                    st.error(f"❌ Erreur : Fichier spécifié non trouvé. Veuillez vérifier le chemin : {e.filename}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement du modèle ou du vectoriseur : {e}")
                    st.stop()
    
                # Nettoyer les textes
                df = data_combined.copy()
                df['Text_cleaned'] = df['Text'].apply(clean_text)
    
                # Transformer les textes nettoyés en vecteurs TF-IDF
                X = vectorizer.transform(df['Text_cleaned']).toarray()
    
                # Obtenir les prédictions du modèle (scores initiaux de 1 à 5)
                predictions = model.predict(X)
                df['Score'] = np.argmax(predictions, axis=1) + 1
    
                # Calculer le score final avec la logique conditionnelle et arrondir par troncature
                df['score_final'] = df.apply(lambda row: calculate_weighted_score(row, alpha, beta), axis=1)
    
                # Attribution du sentiment et de la recommandation
                def assign_sentiment_and_recommendation(row):
                    """
                    Assigne le sentiment et la recommandation en fonction du score final.
                    """
                    if row['score_final'] >= 4:
                        sentiment = 'Positive'
                        recommendation = 'Continue providing excellent service!'
                    elif row['score_final'] == 3:
                        sentiment = 'Neutral'
                        recommendation = 'Consider improving certain aspects to increase satisfaction.'
                    else:
                        sentiment = 'Negative'
                        recommendation = 'Identify issues and work on resolving them to enhance customer experience.'
                    return pd.Series([sentiment, recommendation])
    
                df[['Sentiment', 'Recommendation']] = df.apply(assign_sentiment_and_recommendation, axis=1)
    
                # ------------------------------
                # 🔄 Calcul du Score Global Pondéré (M_p) avec likes/dislikes de la publication
                # ------------------------------
                
                # Définir le facteur de normalisation delta
                delta_normalization = 5  # Maximum possible score
                
                # Calcul de la moyenne globale mu
                mu = df['score_final'].mean()
                
                # Calcul des poids w_i pour chaque score C_i
                # w_i = 1 - |C_i - mu| / delta_normalization
                # Cela donne des poids plus élevés aux scores proches de la moyenne
                # et des poids plus faibles aux scores éloignés de la moyenne
                df['weight'] = 1 - (df['score_final'] - mu).abs() / delta_normalization
                df['weight'] = df['weight'].clip(lower=0)  # S'assurer que les poids sont positifs
                
                # Calcul du score global pondéré M_p avec likes/dislikes de la publication
                numerator = (df['score_final'] * df['weight']).sum() + alpha * st.session_state.total_likes_publication - beta * st.session_state.total_dislikes_publication
                denominator = df['weight'].sum() + alpha + beta
                
                if denominator == 0:
                    st.error("❌ Le dénominateur pour le calcul du score global pondéré est zéro.")
                    st.stop()
                
                adjusted_global_score = numerator / denominator
    
                # S'assurer que le score global est entre 1 et 5 et arrondir par troncature
                adjusted_global_score = math.floor(max(1, min(5, adjusted_global_score)))
    
                # Définir la recommandation globale basée sur le score global ajusté
                if adjusted_global_score in [1, 2]:
                    global_recommendation = "❗️ Urgent améliorations nécessaires pour augmenter la satisfaction des clients."
                elif adjusted_global_score == 3:
                    global_recommendation = "⚠️ Améliorations modérées nécessaires."
                else:  # 4 ou 5
                    global_recommendation = "✅ Bonne performance. Continuez à maintenir les standards."
    
                # Stocker le score global ajusté, la recommandation et le DataFrame analysé dans session_state
                st.session_state.adjusted_global_score = adjusted_global_score
                st.session_state.global_recommendation = global_recommendation
                st.session_state.analyzed_df = df  # Stocker le DataFrame analysé
    
                st.success("✅ Analyse effectuée avec succès!")
    
        # Afficher les détails des commentaires analysés si l'analyse a été effectuée
        if st.session_state.analyzed_df is not None and st.session_state.adjusted_global_score is not None:
            st.markdown("---")
            st.markdown("### 📋 Détails des Commentaires Analyés")
            
            # Sélectionner les colonnes à afficher
            comments_table = st.session_state.analyzed_df[['Text', 'likes', 'dislikes', 'Score', 'score_final']].copy()
            comments_table.rename(columns={
                'Text': 'Commentaire',
                'likes': 'Likes',
                'dislikes': 'Dislikes',
                'Score': 'Score Initial',
                'score_final': 'Score Final'
            }, inplace=True)
        
            # Assurer que 'Likes' et 'Dislikes' sont des entiers
            comments_table['Likes'] = comments_table['Likes'].astype(int)
            comments_table['Dislikes'] = comments_table['Dislikes'].astype(int)
        
            # Fonction pour la mise en forme conditionnelle : Mettre en rouge le score final = 1
            def highlight_final_score(val):
                color = 'background-color: red' if val == 1 else ''
                return color
        
            # Appliquer la mise en forme conditionnelle
            styled_table = comments_table.style.applymap(highlight_final_score, subset=['Score Final'])
        
            # Afficher le tableau stylisé
            st.dataframe(styled_table)
        
            st.markdown("---")
        
            # Bouton pour télécharger les résultats en CSV
            csv = st.session_state.analyzed_df[['Text', 'likes', 'dislikes', 'Score', 'score_final']].copy()
            csv.rename(columns={
                'Text': 'Commentaire',
                'likes': 'Likes',
                'dislikes': 'Dislikes',
                'Score': 'Score Initial',
                'score_final': 'Score Final'
            }, inplace=True)
            csv_download = csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les Résultats en CSV",
                data=csv_download,
                file_name='comments_results.csv',
                mime='text/csv',
            )
    else:
        st.info("ℹ️ Veuillez ajouter des commentaires, importer un fichier, et fournir les données d'engagement de la publication pour lancer l'analyse.")

# ------------------------------
# 5.3. Onglet "Tableau de Bord"
# ------------------------------
with tabs[2]:
    st.subheader("🎨 Tableau de Bord")
    
    # Vérifier si l'analyse a été effectuée
    if st.session_state.analyzed_df is not None and st.session_state.adjusted_global_score is not None:
        df = st.session_state.analyzed_df
        adjusted_global_score = st.session_state.adjusted_global_score
        global_recommendation = st.session_state.global_recommendation
        
        # ------------------------------
        # 10. Affichage des Indicateurs Clés (KPIs) Amélioré
        # ------------------------------
        
        st.markdown("### 📈 Indicateurs Clés de Performance (KPIs)")
        
        # Définir les données des KPIs
        kpis = {
            "📊 Adjusted Global Score": f"{adjusted_global_score}" if adjusted_global_score is not None else "N/A",
            "👍 Total Likes on Comments": f"{df['likes'].sum()}",
            "👎 Total Dislikes on Comments": f"{df['dislikes'].sum()}",
            "👍 Total Likes on Publication": f"{st.session_state.total_likes_publication}",
            "👎 Total Dislikes on Publication": f"{st.session_state.total_dislikes_publication}"
        }
        
        # Définir les couleurs pour chaque KPI
        colors = {
            "📊 Adjusted Global Score": "#4CAF50",  # Vert
            "👍 Total Likes on Comments": "#2196F3",  # Bleu
            "👎 Total Dislikes on Comments": "#f44336",  # Rouge
            "👍 Total Likes on Publication": "#FFC107",  # Jaune
            "👎 Total Dislikes on Publication": "#9C27B0"  # Violet
        }
        
        # Calculer le nombre de colonnes par ligne (par exemple, 3 KPIs par ligne)
        cols_per_row = 3
        rows = (len(kpis) + cols_per_row - 1) // cols_per_row  # Calcul du nombre de lignes nécessaires
        
        # Générer les lignes de KPIs
        kpi_items = list(kpis.items())
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                index = row * cols_per_row + col
                if index < len(kpi_items):
                    kpi_name, kpi_value = kpi_items[index]
                    color = colors.get(kpi_name, "#FFFFFF")  # Couleur par défaut blanche si non défini
                    
                    # HTML/CSS pour styliser chaque KPI avec des tailles réduites
                    kpi_html = f"""
                    <div style="
                        background-color: {color};
                        color: white;
                        padding: 10px;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <h3 style="margin: 0; font-size: 1.2em;">{kpi_name}</h3>
                        <h2 style="margin: 0; font-size: 2em;">{kpi_value}</h2>
                    </div>
                    """
                    cols[col].markdown(kpi_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ------------------------------
        # 11. Génération des Visualisations (Tableau de Bord)
        # ------------------------------
        
        st.markdown("### 📊 Visualisations Interactives")
        
        visualizations = []  # Liste pour stocker les tuples (chemin_image, commentaire)
        
        # Organiser les visualisations en colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Graphique en Barres des Likes et Dislikes Totaux sur les Commentaires
            df_bar = pd.DataFrame({
                'Type': ['Likes', 'Dislikes'],
                'Count': [df['likes'].sum(), df['dislikes'].sum()]
            })
            
            fig1 = px.bar(
                df_bar,
                x='Type',
                y='Count',
                labels={'Type': 'Type', 'Count': 'Nombre'},
                title='🔺 Total Likes et Dislikes sur les Commentaires',
                color='Type',
                color_discrete_map={'Likes': '#2E8B57', 'Dislikes': '#FF6347'},  # Couleurs personnalisées
                hover_data={'Type': True, 'Count': True}
            )
            fig1.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Commentaire pour cette visualisation
            comment1 = "🔺 Ce graphique en barres affiche le nombre total de likes et de dislikes reçus par les commentaires. Un nombre élevé de likes indique une satisfaction générale des clients, tandis que les dislikes peuvent mettre en évidence des domaines à améliorer."
            
            # Sauvegarder la figure en image temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig1.write_image(tmpfile.name)
                visualizations.append((tmpfile.name, comment1))
        
        with col2:
            # 2. Graphique en Secteurs pour la Répartition des Sentiments
            sentiment_counts = df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(
                sentiment_counts,
                names='Sentiment',
                values='Count',
                title='🍰 Répartition des Sentiments',
                color='Sentiment',
                color_discrete_map={'Positive': '#32CD32', 'Neutral': '#808080', 'Negative': '#DC143C'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Commentaire pour cette visualisation
            comment2 = "🍰 Ce graphique en secteurs illustre la répartition des sentiments exprimés dans les commentaires. Une majorité de sentiments positifs indique une haute satisfaction, tandis que les sentiments négatifs mettent en évidence des domaines nécessitant une amélioration."
            
            # Sauvegarder la figure en image temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig_pie.write_image(tmpfile.name)
                visualizations.append((tmpfile.name, comment2))
        
        # Nouvelle ligne de visualisations
        col3, col4 = st.columns(2)
        
        with col3:
            # 3. Treemap pour la Distribution des Scores Finaux
            score_counts = df['score_final'].value_counts().reset_index()
            score_counts.columns = ['Score Final', 'Count']
            fig3 = px.treemap(
                score_counts,
                path=['Score Final'],
                values='Count',
                title='🗺️ Distribution des Scores Finaux',
                color='Count',
                color_continuous_scale='RdBu'
            )
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Commentaire pour cette visualisation
            comment3 = "🗺️ Ce treemap représente la distribution des scores finaux attribués aux commentaires. Il offre une vue intuitive de la fréquence de chaque score, aidant à identifier rapidement les tendances générales dans les évaluations des clients."
            
            # Sauvegarder la figure en image temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig3.write_image(tmpfile.name)
                visualizations.append((tmpfile.name, comment3))
        
        with col4:
            # 4. Histogramme Interactif des Scores
            if pd.api.types.is_numeric_dtype(df['score_final']):
                unique_scores = df['score_final'].unique()
                if len(unique_scores) <= 10:
                    # Utiliser des couleurs discrètes
                    fig_hist = px.histogram(
                        df,
                        x='score_final',
                        nbins=5,
                        title='📊 Distribution des Scores Finaux',
                        labels={'score_final': 'Score Final'},
                        color='score_final',
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                else:
                    # Utiliser des couleurs continues
                    fig_hist = px.histogram(
                        df,
                        x='score_final',
                        nbins=20,
                        title='📈 Distribution des Scores Finaux',
                        labels={'score_final': 'Score Final'},
                        color='score_final',
                        color_continuous_scale='Viridis'
                    )
            else:
                # Si 'score_final' n'est pas numérique, omettre la couleur
                fig_hist = px.histogram(
                    df,
                    x='score_final',
                    nbins=5,
                    title='📉 Distribution des Scores Finaux',
                    labels={'score_final': 'Score Final'}
                )
            
            try:
                fig_hist.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Commentaire pour cette visualisation
                comment4 = "📊 Cet histogramme montre la distribution des scores finaux attribués aux commentaires. Il permet de visualiser la fréquence de chaque score et d'identifier les tendances générales dans les évaluations des clients."
                
                # Sauvegarder la figure en image temporaire
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig_hist.write_image(tmpfile.name)
                    visualizations.append((tmpfile.name, comment4))
            except TypeError as e:
                st.error(f"❌ Erreur lors de la création de l'histogramme : {e}")
                # Optionnel : Afficher un histogramme simple sans couleur
                fig_hist_simple = px.histogram(
                    df,
                    x='score_final',
                    nbins=5,
                    title='📉 Distribution des Scores Finaux',
                    labels={'score_final': 'Score Final'}
                )
                fig_hist_simple.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist_simple, use_container_width=True)
                
                # Commentaire pour l'histogramme simple
                comment4_simple = "📉 Cet histogramme représente la distribution des scores finaux. Il offre une vue simplifiée de la fréquence de chaque score, facilitant l'analyse des tendances générales."
                
                # Sauvegarder l'histogramme simple en image temporaire
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig_hist_simple.write_image(tmpfile.name)
                    visualizations.append((tmpfile.name, comment4_simple))
        
        # ------------------------------
        # 12. Word Cloud
        # ------------------------------
        
        st.markdown("### ☁️ Word Cloud des Commentaires")
        all_text = ' '.join(df['Text_cleaned'])
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
        
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title('☁️ Word Cloud des Commentaires')
        st.pyplot(fig_wc)
        
        # Commentaire pour le Word Cloud
        comment5 = "☁️ Le Word Cloud affiche les mots les plus fréquents présents dans les commentaires. Les mots plus grands apparaissent plus souvent, offrant un aperçu rapide des thèmes récurrents et des préoccupations des clients."
        
        # Sauvegarder le Word Cloud en image temporaire
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig_wc.savefig(tmpfile.name, format='png')
            visualizations.append((tmpfile.name, comment5))
        
        st.markdown("---")
        
        # ------------------------------
        # 13. Génération et Téléchargement du PDF
        # ------------------------------
        
        st.markdown("### 📄 Générer et Télécharger le Rapport PDF")
        
        if st.button("📄 Générer le Rapport PDF"):
            try:
                if not visualizations:
                    st.error("❌ La liste des visualisations est vide. Ajoutez des visualisations avant de générer le PDF.")
                else:
                    pdf_buffer = generate_pdf(visualizations, adjusted_global_score, global_recommendation)
                    st.download_button(
                        label="📥 Télécharger le Rapport PDF",
                        data=pdf_buffer.getvalue(),  # Obtenir les bytes du buffer
                        file_name='comments_report.pdf',
                        mime='application/pdf',
                    )
                    st.success("✅ Rapport PDF généré avec succès!")
                    
                    # Nettoyer les fichiers temporaires automatiquement
                    for img_path, _ in visualizations:
                        try:
                            os.remove(img_path)
                        except Exception as e:
                            st.warning(f"⚠️ Impossible de supprimer le fichier temporaire {img_path} : {e}")
            except ValueError as ve:
                st.error(f"❌ Erreur lors de la génération du PDF : {ve}")
            except Exception as e:
                st.error(f"❌ Une erreur est survenue lors de la génération du PDF : {e}")
        
        # ------------------------------
        # 14. Affichage du Résumé des Résultats
        # ------------------------------
        
        st.markdown("---")
        st.markdown("### 📋 Résumé des Résultats")
        st.write(f"**📊 Adjusted Global Score :** {adjusted_global_score}")
        st.write(f"**📝 Recommandation :** {global_recommendation}")
    
    else:
        st.info("ℹ️ Veuillez lancer l'analyse en ajoutant des commentaires et en fournissant les données d'engagement de la publication.")
