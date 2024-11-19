def fetch_submission_data(reddit_client, submission_id):
    """Obtiene un submission específico dado un ID"""
    # Obtener el submission dado el ID
    submission = reddit_client.submission(id=submission_id)

    # Obtener todos los comentarios
    submission.comments.replace_more(limit=None)
    comments = submission.comments.list()

    # Información del submission
    submission_info = {
        "id": submission.id,
        "title": submission.title,
        "score": submission.score,
        "url": submission.permalink,
        "created": submission.created,
        "comments": [
            {
                "comment_id": comment.id,
                "author": comment.author.name if comment.author else None,
                "comment": comment.body,
                "created": comment.created,
            }
            for comment in comments
        ],
    }
    return submission_info


def fetch_submissions_by_topic(reddit_client, topic, limit):
    """Obtiene una lista de submissions dado un topic"""
    # Filtrar por topic
    subreddit = reddit_client.subreddit(topic)
    submissions = []
    for submission in subreddit.new(limit=limit):
        # Obtener los comentarios del subreddit junto con las respuestas
        submission.comments.replace_more(limit=None)
        comments = submission.comments.list()

        # Añadir a la lista de submissions
        submissions.append(
            {
                "id": submission.id,
                "title": submission.title,
                "score": submission.score,
                "url": submission.permalink,
                "created": submission.created,
                "comments": [
                    {
                        "comment_id": comment.id,
                        "author": comment.author.name if comment.author else None,
                        "comment": comment.body,
                        "created": comment.created,
                    }
                    for comment in comments
                ],
            }
        )
    return submissions


def fetch_comments_by_query(reddit_client, query, after, before, size):
    """
    Obtiene comentarios de Reddit con una búsqueda específica para un periodo de tiempo dado.

    Parameters:
    - reddit_client: Reddit Client (praw)
    - query: La búsqueda de términos (por ejemplo, "women are secretary")
    - after: Timestamp de inicio
    - before: Timestamp de fin
    - size: Número de comentarios a recuperar
    """
    # Crear lista para almacenar los comentarios
    comments_list = []

    # Realizar la búsqueda
    for submission in reddit_client.subreddit("all").comments(
        limit=size, params={"q": query, "before": before, "after": after}
    ):
        comments_list.append({"id": submission.id, "comments": submission.body})

    return comments_list
