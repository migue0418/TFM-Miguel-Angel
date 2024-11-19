import torch
import math


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
    Gets perplexities of all sentences in a DataFrame(contains 2 columns of contrasting sentences)
    based on given model
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
