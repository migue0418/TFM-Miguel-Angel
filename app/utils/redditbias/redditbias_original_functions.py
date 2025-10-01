import torch
import math
import numpy as np
import re


def perplexity_score(sentence, model, tokenizer):
    """
    Finds perplexity score of a sentence based on model
    Parameters
    ----------
    sentence : str
    Given sentence
    model :
    Pre-trained language model
    tokenizer :
    Pre-trained tokenizer

    Returns
    -------
    Perplexity score
    """
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0])


def model_perplexity(sentences, model, tokenizer):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    sentences : list
    sentence set
    model :
    Pre-trained language model
    tokenizer :
    Pre-trained tokenizer

    Returns
    -------
    Model perplexity score
    """
    total_loss = 0
    for sent in sentences:
        with torch.no_grad():
            model.eval()
            tokenize_input = tokenizer.tokenize(sent)
            tensor_input = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenize_input)]
            )
            loss = model(tensor_input, labels=tensor_input)
            total_loss += loss[0]
    return math.exp(total_loss / len(sentences))


def get_perplexity_list(df, m, t):
    """
    Gets perplexities of all sentences in a DataFrame based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = perplexity_score(row["comments_processed"], m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    """
    Gets perplexities of all sentences in a DataFrame(contains 2 columns
    of contrasting sentences) based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments in 2 columns
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == "black":
                perplexity = perplexity_score(row["comments_1"], m, t)
            else:
                perplexity = perplexity_score(row["comments_2"], m, t)
        except Exception:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def process_tweet(sent):
    """
    Pre-processes a given sentence
    Parameters
    ----------
    sent : str
    Given sentence

    Returns
    -------
    Processed sentence
    """

    sent = sent.encode("ascii", errors="ignore").decode()  # check this output
    # print(sent)
    sent = re.sub(r"@[^\s]+", "", sent)
    sent = re.sub(r"https: / /t.co /[^\s]+", "", sent)
    sent = re.sub(r"http: / /t.co /[^\s]+", "", sent)
    sent = re.sub(r"http[^\s]+", "", sent)

    sent = re.sub(r"&gt", "", sent)

    # split camel case combined words
    sent = re.sub(r"([A-Z][a-z]+)", r"\1", re.sub("([A-Z]+)", r" \1", sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(r" \d+", "", sent)
    # remove words with letter+number
    sent = re.sub(r"\w+\d+|\d+\w+", "", sent)

    # remove spaces
    sent = re.sub(r"[\s]+", " ", sent)
    sent = re.sub(r"[^\w\s.!\-?]", "", sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(r" rt ", "", sent)

    sent = re.sub(r"- ", "", sent)

    sent = sent.strip()
    # print(sent)
    return sent


def build_dataset_bos_eos(df, demo, dest_path):
    """
    Writes data from Dataframe to a text file, each dataframe row line by line in text
    file appending BOS and EOS token
    Parameters
    ----------
    df : pd.DataFrame
    Dataframe of biased reddit phrases
    demo : str
    Demographic name
    dest_path : str
    Path to store text file

    """
    with dest_path.open("w") as f:
        data = ""

        for idx, row in df.iterrows():
            bos_token = "<bos>"
            eos_token = "<eos>"
            comment = row["comments_2"]
            data += bos_token + " " + comment + " " + eos_token + "\n"

        f.write(data)


def build_dataset_manual_annot(df, demo, dest_path, reduced: bool = False):
    """
    Writes data from Dataframe to a text file, each dataframe row line by line in text file
    Parameters
    ----------
    df : pd.DataFrame
    Dataframe of biased reddit phrases
    demo : str
    Demographic name
    dest_path : str
    Path to store text file

    """
    with dest_path.open("w") as f:
        data = ""

        for idx, row in df.iterrows():
            comment = row["comments_processed"]
            if demo == "orientation" and reduced:
                data += "<bos>" + " " + comment + "\n"
            else:
                data += comment + "\n"

        f.write(data)


def replace_with_caps(text, replacements):
    for i, j in replacements.items():
        text = text.replace(i, j)
    return text


def get_model_perplexity(df, m, t):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    Model perplexity
    """
    model_perp = model_perplexity(df["comments_processed"], m, t)
    return model_perp


def find_anomalies(data):
    """
    Find outliers in a given data distribution
    Parameters
    ----------
    data : list
    List of sentence perplexities

    Returns
    -------
    List of outliers
    """
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies
