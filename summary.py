from summar_chunking import summary_chunking
from generate_summary import generate_summary

def summary(file_name, percentile=90, dynamic_threshold=50, dynamic_adjustment = False):
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
        content_list = summary_chunks(file_name=file_name, percentile=percentile)
        total_chunk = int(len(content_list)*(100/(100-percentile)))
        print(f"total chunk: {total_chunk} chunks selected: {len(content_list)}")
        curr_delta = len(content_list) - dynamic_threshold
        updated_percentile = percentile + (curr_delta/total_chunk)*100
    print(f"Updated Percentile: {updated_percentile}")
    content_list = summary_chunks(file_name=file_name, percentile=updated_percentile)

    context = "\n".join(content_list)
    print(f"Selected chunks: {len(content_list)}")
    response = generate_summary(context)
    
    end = time.time()
    print("Response: ",response, "\nTime:",end-start)

    return response
