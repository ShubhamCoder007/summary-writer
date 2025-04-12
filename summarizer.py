import utils.set_log_config as lg
from config_file import config
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from model import llm
from utils.document_retrieval import Document_Retrieval
from utils.log_utils import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from prompts.summary_prompt import summary_prompt
from sklearn.metrics.pairwise import cosine_similarity
from summary_test import boosted_tfidf_vector
from prompt import task
logger = Logger()
import time


@logger.log_start_end
def boosted_tfidf_vector(chunks, query, boost_factor=3.0, stop_words='english'):
    """
    Vectorizes document chunks and query using TF-IDF (with stop word filtering).
    Boosts the chunk vectors to emphasize tokens that appear in the query, then computes cosine similarities.
    
    Args:
        chunks (list of str): List of document chunks.
        query (str): The query text.
        boost_factor (float): The factor by which to boost chunk vector components that match query tokens.
        stop_words (str or list): Stop words to remove during vectorization.
        
    Returns:
        numpy.array: Cosine similarity scores between each boosted chunk vector and the query vector.
        chunk vector
    """
    # Initialize and fit the TF-IDF vectorizer on the chunks
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    # Transform the query with the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Retrieve feature names and create a boost mask for features that are in the query
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Simple splitting on whitespace; consider enhancements for more robust tokenization
    query_terms = query.split()
    boost_mask = np.isin(feature_names, query_terms)  # Boolean array, True if the token is in the query
    
    # Define boost weights: boost_factor for features in the query, 1.0 otherwise
    boost_weights = np.where(boost_mask, boost_factor, 1.0)
    
    # Convert sparse matrices to dense arrays for element-wise operations
    chunk_vectors_dense = chunk_vectors.toarray()
    query_vector_dense = query_vector.toarray()  # Not boosted, keeps the query representation unchanged
    
    # Boost the chunk vectors: multiply each feature column with the corresponding boost weight
    boosted_chunk_vectors = chunk_vectors_dense * boost_weights  # Broadcasting boost_weights along rows

    # print(chunk_vectors_dense[0],"\n\n", boosted_chunk_vectors[0])
    
    # Compute cosine similarity between each boosted chunk vector and the query vector
    similarities = cosine_similarity(boosted_chunk_vectors, query_vector_dense)
    
    return similarities, boosted_chunk_vectors


@logger.log_start_end
def summary_chunks(file_name, percentile, summary_query=""):
    ob = Document_Retrieval(index_name=config["azure"]["ai_search_index"])
    filter_ = f"source eq 'para' and file_name eq '{file_name}'"
    result_ = list(ob.single_vector_search_with_filter_test(" ", "content_vector", filter_, None, 10000, 10000))
    print(f"Total {len(result_)} chunks found for {file_name}")
    result_ = sorted(result_, key=lambda x: x['seq_num'])
    chunk_list = [content if res is not None and (content := res['content']) is not None else "none" for res in result_]
    
    if summary_query=="":
        print("Simple tfidf vectorization")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(chunk_list)
    else:
        print("Boosted tfidf vectorization")
        _, X = boosted_tfidf_vector(chunks=chunk_list, query=summary_query, boost_factor=2)
    row_sums = np.array(X.sum(axis=1)).flatten()

    percentile = np.percentile(row_sums, percentile)
    content_list = [content for content, score in zip(chunk_list, row_sums) if score >= percentile]

    return content_list


@logger.log_start_end
def generate_summary(context):
    prompt_summary = summary_prompt
    prompt_qd = PromptTemplate(template=prompt_summary, input_variables=["context"])
    chain_qd = LLMChain(llm=llm, prompt=prompt_qd)
    response = chain_qd.invoke(input={"context": context})
    response = response["text"]
    return response


@logger.log_start_end
def summary(file_name, percentile=90, dynamic_threshold=50, dynamic_adjustment = False, summary_query=""):
    """
    Generate a summary for the given file.

    Args:
        file_name (str): The name of the file to summarize.
        percentile (int, optional): The percentile threshold for selecting chunks. Default is 90.

    Returns:
        str: The generated summary.
    """
    start = time.time()
    logger.add_common_extra(lg.log_config["properties"])

    updated_percentile=percentile
    if dynamic_adjustment:
        content_list = summary_chunks(file_name=file_name, percentile=percentile, summary_query=summary_query)
        total_chunk = int(len(content_list)*(100/(100-percentile)))
        print(f"total chunk: {total_chunk} chunks selected: {len(content_list)}")
        curr_delta = len(content_list) - dynamic_threshold
        updated_percentile = percentile + (curr_delta/total_chunk)*100
    print(f"Updated Percentile: {updated_percentile}")
    content_list = summary_chunks(file_name=file_name, percentile=updated_percentile, summary_query=summary_query)

    context = "\n".join(content_list)
    print(f"Selected chunks: {len(content_list)}")
    response = generate_summary(context)
    
    end = time.time()
    print("Response: ",response, "\nTime:",end-start)

    return response


# Usage
summary(
    file_name = "bowlero",
    percentile=90,
    dynamic_threshold=20,
    dynamic_adjustment=False,
    summary_query="revenue risk"
    )


