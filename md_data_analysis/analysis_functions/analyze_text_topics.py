import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk
from typing import Optional

try:
    STOPWORDS_PT = stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    STOPWORDS_PT = stopwords.words('portuguese')

def analyze_text_topics(df: pd.DataFrame, bairro: Optional[str] = None, n_topics: int = 5, n_keywords: int = 7):
    df_filtrado = df.copy()
    if bairro:
        df_filtrado = df_filtrado[df_filtrado['bairro'].str.contains(bairro, case=False, na=False)]

    text_data = df_filtrado['descricao_modus_operandi'].dropna()

    if len(text_data) < 10:
        return {"message": "Texto insuficiente para modelagem de tópicos com os filtros aplicados.", "topics": []}
    vectorizer = CountVectorizer(
        max_df=0.9, 
        min_df=2, 
        stop_words=STOPWORDS_PT,
        lowercase=True
    )
    X = vectorizer.fit_transform(text_data)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-n_keywords - 1:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topics.append({
            "topic_id": topic_idx,
            "keywords": top_keywords
        })
        
    return {
        "message": f"{len(topics)} tópicos principais encontrados no modus operandi.",
        "topics": topics
    }