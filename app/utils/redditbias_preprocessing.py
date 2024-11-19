import concurrent.futures
import pandas as pd
from app.external_services.reddit_client import get_reddit_client
from app.schemas.redditbias import Topic
from app.utils.reddit_fetcher import fetch_comments_by_query
from core.config import data_path
from fastapi import HTTPException


def get_raw_reddit_comments(topic_instance: Topic, size: int, chunks: int):
    # Define input file paths
    query_feature_path = (
        data_path
        / topic_instance.name
        / f"{topic_instance.name}_{topic_instance.minority_group}.txt"
    )
    query_demo_path = (
        data_path / topic_instance.name / f"{topic_instance.name}_opposites.txt"
    )

    # Check if input files exist
    if not query_feature_path.exists() or not query_demo_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Read the contents of input files
    with open(query_feature_path) as f:
        query_feature = [line.strip() for line in f]

    with open(query_demo_path) as f:
        query_demo = [line.strip().split(",")[0] for line in f]

    # Determine the number of loops required (process up to 4 queries per loop)
    loops = (len(query_demo) + 3) // 4

    # Create the output directory if it doesn't exist
    output_dir = data_path / topic_instance.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the Reddit client
    reddit_client = get_reddit_client(user_agent="get_reddit_posts")

    # Process comments in batches using threads
    for i in range(loops):
        comments_list = []
        query_demo_4 = query_demo[i * 4 : (i + 1) * 4]

        # Use ThreadPoolExecutor to fetch comments concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks to the executor for each query
            future_to_query = {
                executor.submit(
                    fetch_comments_by_query,
                    reddit_client,
                    qd,
                    query_feature,
                    size,
                    chunks,
                ): qd
                for qd in query_demo_4
            }

            # Process results as tasks complete
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    comments_list += future.result()
                except Exception as e:
                    print(
                        f"Error! Could not obtain the comments for {future_to_query[future]}: {e}"
                    )

        # Convert the list of comments to a DataFrame
        comments_df = pd.DataFrame(comments_list, columns=["id", "comments"])

        # Define and saved the result CSV file
        output_file = (
            output_dir
            / f"reddit_comments_{topic_instance.name}_{topic_instance.minority_group}_raw_{i}.csv"
        )
        comments_df.to_csv(output_file)

        print(f"Successfully stored CSV file: {output_file}")
