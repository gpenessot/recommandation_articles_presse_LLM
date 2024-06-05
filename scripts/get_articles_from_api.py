from newsapi import NewsApiClient
from datetime import datetime as dt
import configparser
import pandas as pd
import glob
import os

def get_news_api_client() -> NewsApiClient:
    """
    Initialize NewsApiClient with API key from environment variable or config file.

    Returns:
        NewsApiClient: An instance of NewsApiClient initialized with the API key.
    """
    api_key = os.environ.get('NEWSAPI_TOKEN')
    if not api_key:
        conf = configparser.ConfigParser()
        conf.read('../config/config.cfg')
        api_key = conf.get('newsapi', 'key')
    return NewsApiClient(api_key=api_key)

def fetch_articles(newsapi, sources_fr, pages=5):
    """
    Fetch articles from specified sources for multiple pages.

    Args:
        newsapi (NewsApiClient): Initialized NewsApiClient object.
        sources_fr (list): List of French news sources.
        pages (int): Number of pages to fetch (default is 5).

    Returns:
        pd.DataFrame: Concatenated DataFrame of fetched articles.
    """
    concat_articles = pd.DataFrame()
    for p in range(1, pages+1):
        try:
            top_headlines = newsapi.get_top_headlines(sources=', '.join(sources_fr),
                                                      page=p,
                                                      language='fr')
            df_articles_fr = pd.DataFrame().from_dict(top_headlines['articles'])
            concat_articles = pd.concat([concat_articles, df_articles_fr])
        except Exception as e:
            print(f'Error on page {p}: {e}')
            continue
    return concat_articles

def clean_combined_csv(combined_csv):
    """
    Clean combined CSV DataFrame.

    Args:
        combined_csv (pd.DataFrame): Combined CSV DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    combined_csv = combined_csv.dropna(subset=['content', 'author'])
    combined_csv = combined_csv[~combined_csv['content'].str.startswith('Search\r\nDirect')]
    combined_csv['publishedAt'] = pd.to_datetime(combined_csv['publishedAt']).dt.tz_localize(None)
    combined_csv = combined_csv.reset_index(drop=True)
    col_to_keep = ['author', 'title', 'publishedAt', 'content']
    combined_csv = combined_csv[col_to_keep]
    dtype_dict = {'author':'str',
                  'title':'str',
                  'publishedAt':'str',
                  'content':'str'}
    combined_csv = combined_csv.astype(dtype_dict)
    combined_csv = combined_csv.drop_duplicates()
    return combined_csv

def main():
    """Main function to execute the data retrieval and processing."""
    newsapi = get_news_api_client()

    liste_sources = newsapi.get_sources()
    df_sources = pd.DataFrame().from_dict(liste_sources['sources'])
    sources_fr = df_sources[df_sources['country'] == 'fr']['id'].tolist()
    sources_fr.pop(2)  # Remove the third source

    concat_articles = fetch_articles(newsapi, sources_fr)

    # Absolute path for saving raw data
    concat_articles.to_csv(f'/home/runner/work/recommandation_articles_presse_LLM/recommandation_articles_presse_LLM/data/raw/news_{dt.now().month}-{dt.now().day}.csv')

    # Use absolute path for glob
    csv_files = glob.glob('/home/runner/work/recommandation_articles_presse_LLM/recommandation_articles_presse_LLM/data/raw/*.csv')
    combined_csv = pd.concat((pd.read_csv(file, index_col=0) for file in csv_files))

    combined_csv = clean_combined_csv(combined_csv)

    # Absolute path for saving processed data
    combined_csv.to_csv('/home/runner/work/recommandation_articles_presse_LLM/recommandation_articles_presse_LLM/data/processed/articles.csv')

if __name__ == "__main__":
    main()
