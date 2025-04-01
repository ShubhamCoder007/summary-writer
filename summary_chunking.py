from config_file import config
from utils.document_retrieval import Document_Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from prompts.summary_prompt import summary_prompt

def summary_chunks(file_name, percentile):
    ob = Document_Retrieval(index_name=config["azure"]["ai_search_index"])
    filter_ = f"source eq 'para' and file_name eq '{file_name}'"
    result_ = list(ob.single_vector_search_with_filter_test(" ", "content_vector", filter_, None, 10000, 10000))
    print(f"Total {len(result_)} chunks found for {file_name}")
    result_ = sorted(result_, key=lambda x: x['seq_num'])
    chunk_list = [content if res is not None and (content := res['content']) is not None else "none" for res in result_]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunk_list)
    row_sums = np.array(X.sum(axis=1)).flatten()

    percentile = np.percentile(row_sums, percentile)
    content_list = [content for content, score in zip(chunk_list, row_sums) if score >= percentile]

    return content_list
