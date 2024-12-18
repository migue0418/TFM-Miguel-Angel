import concurrent.futures
from typing import Literal
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from app.external_services.reddit_client import get_reddit_client
from app.schemas.redditbias import Topic
from app.utils.reddit_fetcher import fetch_comments_by_query
from app.utils.redditbias_original_functions import (
    build_dataset_manual_annot,
    process_tweet,
)
from app.core.config import (
    redditbias_data_path as data_path,
    redditbias_files_path as files_path,
)
from fastapi import HTTPException


def get_raw_reddit_comments(topic_ins: Topic, size: int, chunks: int):
    # Define input file paths
    query_feature_path = (
        data_path / topic_ins.name / f"{topic_ins.name}_{topic_ins.minority_group}.txt"
    )
    query_demo_path = data_path / topic_ins.name / f"{topic_ins.name}_opposites.txt"

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
    output_dir = data_path / topic_ins.name
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
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}_raw_{i}.csv"
        )
        comments_df.to_csv(output_file)

        print(f"Successfully stored CSV file: {output_file}")


def reddit_data_process(topic_ins: Topic, process_demo1: bool = True):
    """
    This script processes the raw Reddit comments for Target group 1(from reddit_data.py)
    and further creates Counter Target dataset containing target group term replaced with
    Target group 2 terms. In case of demographic - Gender and Sexual orientation,
    Reddit comments with only one Target group mention are retained.
    """
    # Set the path for the processed file
    processed_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}_processed.csv"
    )
    processed_majority_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}_processed.csv"
    )

    # Process Reddit comments in all raw files and store in processed file for Target group 1
    if process_demo1:
        print("Processing demo1 reddit files...")
        colNames = ("id", "comments", "comments_processed")

        demo1_df_processed = pd.DataFrame(columns=colNames)
        df_list = []
        if topic_ins.name == "gender" or topic_ins.name == "religion2":
            loops = 7
        elif topic_ins.name == "race" or topic_ins.name == "orientation":
            loops = 5
        elif topic_ins.name == "religion1":
            loops = 6
        else:
            loops = None
            print("Specify a correct demographic")

        for i in range(loops):
            raw_file_path = (
                data_path / topic_ins.name / f"reddit_comments_{topic_ins.name}_"
                + f"{topic_ins.minority_group}_raw_{i}.csv"
            )
            demo1_df = pd.read_csv(raw_file_path)

            demo1_df = demo1_df.loc[:, ~demo1_df.columns.str.contains("^Unnamed")]

            demo1_df = demo1_df.dropna()

            demo1_df["comments_processed"] = demo1_df["comments"].apply(
                lambda x: process_tweet(x)
            )
            print("Before length filter {}".format(demo1_df.shape))
            demo1_df = demo1_df[demo1_df["comments_processed"].str.len() < 150]
            # pd.concat([demo1_df_processed, demo1_df])
            print("After length filter {}".format(demo1_df.shape))
            # demo1_df_processed.append(demo1_df, ignore_index=True)
            df_list.append(demo1_df)

        demo1_df_processed = pd.concat(df_list, ignore_index=True)
        print(demo1_df_processed.shape)
        demo1_df_processed = demo1_df_processed.dropna()
        demo1_df_processed = demo1_df_processed[
            demo1_df_processed["comments_processed"] != "nan"
        ]
        print("After dropping nan {}".format(demo1_df_processed.shape))

        demo1_df_processed.to_csv(processed_file_path, index=False)

    # If the topic is gender or orientation retain sentences with only one target group term
    if topic_ins.name == "gender":
        colNames = ("id", "comments_processed")
        demo2_df = pd.DataFrame(columns=colNames)

        gender_words = [
            "woman",
            "women",
            "girl",
            "mother",
            "daughter",
            "wife",
            "niece",
            "mom",
            "bride",
            "lady",
            "madam",
            "hostess",
            "female",
            "wife",
            "aunt",
            "sister",
            "man",
            "men",
            "boy",
            "father",
            "son",
            "husband",
            "nephew",
            "dad",
            "groom",
            "gentleman",
            "sir",
            "host",
            "male",
            "husband",
            "uncle",
            "brother",
        ]
        comments_one_g = []
        for idx, row in demo1_df_processed.iterrows():
            s = row["comments_processed"]
            match = {m for m in gender_words if m in s}
            print(match)
            if len(match) == 1:
                comments_one_g.append(s)
        demo2_df["comments_processed"] = comments_one_g
        print("Shape of df with single target group comments {}".format(demo2_df.shape))
        demo1_df_processed = demo2_df
        demo1_df_processed.to_csv(processed_file_path, index=False)

    if topic_ins.name == "orientation":
        colNames = ("id", "comments_processed")
        demo2_df = pd.DataFrame(columns=colNames)

        orientation_words = [
            "gay",
            "lesbian",
            "bisexual",
            "homosexual",
            "transgender",
            "sapphic",
            "pansexual",
            "queer",
        ]
        comments_one_g = []
        for idx, row in demo1_df_processed.iterrows():
            s = row["comments_processed"]
            match = {m for m in orientation_words if m in s}
            print(match)
            if len(match) == 1:
                comments_one_g.append(s)
        demo2_df["comments_processed"] = comments_one_g
        print("Shape of df with single target group comments {}".format(demo2_df.shape))
        demo1_df_processed = demo2_df
        demo1_df_processed.to_csv(processed_file_path, index=False)
    else:
        print("Reading processed demo1 reddit files...")
        demo1_df_processed = pd.read_csv(processed_file_path)
        print("Shape of demo1 data {}".format(demo1_df_processed.shape))

    # Create Counter target data set
    demo2_df = pd.DataFrame(
        columns=["initial_demo", "replaced_demo", "comments", "comments_processed"]
    )

    if topic_ins.name == "race":
        pairs = (
            ("black", "white"),
            ("african american", "anglo american"),
            ("african-american", "anglo-american"),
            ("afro-american", "anglo-american"),
            ("african", "american"),
            ("afroamericans", "angloamericans"),
            ("negroes", "caucasians"),
            ("dark-skin", "light-skin"),
            ("dark skin", "light skin"),
        )
    elif topic_ins.name == "religion1":
        pairs = (
            ("jew ", "christian "),
            ("jewish", "christian"),
            ("jews ", "christians "),
            ("judaism", "christianity"),
        )
    elif topic_ins.name == "religion2":
        pairs = (
            ("muslim", "christian"),
            ("islamic", "christian"),
            ("islam ", "christianity "),
            ("arabs", "americans"),
            ("islamism", "christianity"),
        )
    elif topic_ins.name == "gender":
        pairs = (
            ("woman", "man"),
            ("women", "men"),
            ("girl", "boy"),
            ("mother", "father"),
            ("daughter", "son"),
            ("wife", "husband"),
            ("niece", "nephew"),
            ("mom", "dad"),
            ("bride", "groom"),
            ("lady", "gentleman"),
            ("madam", "sir"),
            ("hostess", "host"),
            ("female", "male"),
            ("wife", "husband"),
            ("aunt", "uncle"),
            ("sister", "brother"),
            (" she ", " he "),
        )
    else:
        pairs = (
            ("gay", "straight"),
            ("gays", "straights"),
            ("lesbian", "straight"),
            ("lesbians", "straights"),
            ("bisexual", "monosexual"),
            ("bisexuals", "monosexuals"),
            ("homosexual", "heterosexual"),
            ("homosexuals", "heterosexuals"),
            ("transgender", "cisgender"),
            ("transgenders", "cisgenders"),
            ("sapphic", "heterosexual"),
            ("pansexual", "heterosexual"),
            ("queer", "heterosexual"),
        )

    for idx, row in demo1_df_processed.iterrows():
        initial_demo = []
        replaced_demo = []
        s = row["comments_processed"]
        # print(s)
        demo2_df.at[idx, "comments"] = s

        for p in pairs:
            # s = s.replace(*p)
            if topic_ins.name == "race":
                if (
                    p[0] == "african"
                    and p[0] in s
                    and ("anglo american" in s or "anglo-american" in s)
                ):
                    s = s.replace(*p)
                elif (
                    p[1] == "american"
                    and p[1] in s
                    and ("anglo american" in s or "anglo-american" in s)
                ):
                    s = s.replace(*p)
                elif p[0] == "afro-american" and p[0] in s:
                    s = s.replace(*p)
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "religion1":
                if p[0] == "jewish":
                    if p[0] in s and ("christian" in s):
                        s = s.replace(*p)
                    elif "christian" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "religion2":
                if p[0] == "islamic":
                    if p[0] in s and ("christian" in s):
                        s = s.replace(*p)
                    elif "christian" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                elif p[0] == "islamism":
                    if p[0] in s and ("christianity" in s):
                        s = s.replace(*p)
                    elif "christianity" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "gender":
                s = s.replace(*p)
            elif topic_ins.name == "orientation":
                s = s.replace(*p)

            if p[1] in s and p[0] in row["comments_processed"]:
                initial_demo.append(p[0])
                replaced_demo.append(p[1])
        demo2_df.at[idx, "comments_processed"] = s
        demo2_df.at[idx, "initial_demo"] = initial_demo
        demo2_df.at[idx, "replaced_demo"] = replaced_demo

    print("Shape of demo2 data {}".format(demo2_df.shape))
    demo2_df.to_csv(processed_majority_file_path, index=False)


def reddit_data_phrases(topic_ins: Topic, remove_no_attribute_in_window: bool):
    """
    This script generates phrases from processed Reddit comments such that each phrase is
    maximum length of 15 and contains target group term and attribute term
    """
    # Define the files paths
    processed_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}_processed.csv"
    )
    topic_txt_file_path = (
        data_path / topic_ins.name / f"{topic_ins.name}_{topic_ins.minority_group}.txt"
    )
    processed_phrase_file_path = (
        data_path / topic_ins.name / f"reddit_comments_{topic_ins.name}_"
        + f"{topic_ins.minority_group}_processed_phrase.csv"
    )

    demo1_df_processed = pd.read_csv(processed_file_path)
    print(demo1_df_processed.shape)

    targets = []
    attributes = []

    # Since targets in 'demo_opposites.txt'(ex: race_opposites.txt) are phrases('africans are'),
    # here the targets are listed separately
    if topic_ins.name == "race":
        targets = [
            "black",
            "blacks",
            "african",
            "africans",
            "afro-americans",
            "negroes",
            "dark-skinned",
            "african-americans",
        ]
        with open(topic_txt_file_path) as f:
            attributes = [re.sub('[*"]', "", line.split("\n")[0]) for line in f]
        print(attributes)
    elif topic_ins.name == "gender":
        targets = [
            "women",
            "mothers",
            "woman",
            "girl",
            "wife",
            "niece",
            "mom",
            "moms",
            "grandmother",
            "stepdaughter",
            "bride",
            "lady",
            "madam",
            "granddaughter",
            "hostess",
            "girlfriend",
            "females",
            "wives",
            "aunt",
            "sisters",
            "sister",
            "girlfriends",
        ]
        with open(topic_txt_file_path) as f:
            attributes = [re.sub('[*"]', "", line.split("\n")[0]) for line in f]
        print(attributes)
    elif topic_ins.name == "religion1":
        targets = ["jew", "Jews", "Jewish", "Torah", "Judaism", "Semitic", "Ashkenazi"]
        with open(topic_txt_file_path) as f:
            attributes = [re.sub('[*"]', "", line.split("\n")[0]) for line in f]
        print(attributes)
    elif topic_ins.name == "religion2":
        targets = ["muslim", "muslims", "islam", "islamic", "arab", "arabs"]
        with open(topic_txt_file_path) as f:
            attributes = [re.sub('[*"]', "", line.split("\n")[0]) for line in f]
        print(attributes)
    elif topic_ins.name == "orientation":
        targets = [
            "gay",
            "gays",
            "lesbian",
            "lesbians",
            "bisexual",
            "bisexuals",
            "homosexual",
            "homosexuals",
            "transgender",
            "transgenders",
            "sapphic",
            "pansexual",
            "pansexuals",
            "queer",
            "queers",
        ]
        with open(topic_txt_file_path) as f:
            attributes = [re.sub('[*"]', "", line.split("\n")[0]) for line in f]
        print(attributes)

    data_list = []

    for idx, row in demo1_df_processed.iterrows():
        row_dict = {}
        phrase_joined = ""
        sent = row["comments_processed"]
        try:
            sent_list = sent.split(" ")
            print(sent_list)
            targets_in_sent = [t.lower() for t in targets if t.lower() in sent_list]
            print(targets_in_sent)
            # if len(targets_in_sent) == 0:
            #     print(sent)
            for target in targets_in_sent:
                # print(target)
                # target = random.choice(targets_in_sent)

                target_index1, target_index2 = None, None
                target_index1 = sent_list.index(target.strip())

                # print(target_index1)
                # print(sent_list.count(target))

                if sent_list.count(target) > 1:
                    sent_list_2 = sent_list[target_index1 + 1 :]
                    # print('Sentence 2 is {}'.format(sent_list_2))
                    target_index2 = sent_list_2.index(target.strip())
                    target_index2 = target_index1 + 1 + target_index2

                # print(target_index1, target_index2)

                # If the sentence has two mentions of target group term, select the phrase
                # (cropped sentence) that contains attribute term
                for target_index in [target_index1, target_index2]:

                    if target_index is not None:
                        left_window, right_window = (
                            target_index - 7,
                            target_index + 7 + 1,
                        )

                        if left_window < 0:
                            left_window = 0
                        phrase_list = sent_list[left_window:right_window]
                        phrase_joined = " ".join(phrase_list)

                        # Extract the phrase if any of thr pre-defined attributes are in it
                        if any(attr.lower() in phrase_joined for attr in attributes):
                            row_dict["id"] = row["id"]
                            row_dict["attribute_in_window"] = True
                            row_dict["comment"] = row["comments_processed"]
                            row_dict["phrase"] = phrase_joined
                            data_list.append(row_dict)
                            break

            if not row_dict:
                row_dict["id"] = row["id"]
                row_dict["attribute_in_window"] = False
                row_dict["comment"] = row["comments_processed"]
                row_dict["phrase"] = phrase_joined
                data_list.append(row_dict)

        except Exception:
            pass

    data_df = pd.DataFrame(data_list)
    print(data_df.shape)
    data_df = data_df.drop_duplicates(subset=["phrase"])
    print(data_df.shape)

    if remove_no_attribute_in_window:
        data_df = data_df[data_df.attribute_in_window]

    print(data_df.shape)

    data_df.to_csv(processed_phrase_file_path, index=False)


def reddit_reduce_for_annotation(topic_ins: Topic):
    """
    In this script, phrases with attributes related to career and interests are retained
    from the earlier extracted Reddit phrases for demographic Gender
    """
    suffix_file = "_processed_phrase"
    processed_phrase_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix_file}.csv"
    )
    suffix_file = "_processed_phrase_for_annot"
    processed_anot_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix_file}.csv"
    )
    demo1_processed_phrase = pd.read_csv(processed_phrase_file_path)
    print(demo1_processed_phrase.shape)

    if topic_ins.name == "gender":
        data_career = []
        career_attributes = [
            "nurse",
            "secretary",
            "housekeep",
            "cook",
            "sewing",
            "cleaning",
            "waitress",
            "maid",
            "child-care",
            "receptionist",
            "clerk",
            "home-care",
        ]
        for idx, row in demo1_processed_phrase.iterrows():
            # print(row.to_dict())
            if any(attr in row["phrase"] for attr in career_attributes):
                # print('yes')
                data_career.append(row.to_dict())

        print("Career dataset shape {}".format(len(data_career)))
        suffix_file = "_processed_phrase_extra"
        extra_file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix_file}.csv"
        )
        demo1_processed_phrase_extra = pd.read_csv(extra_file_path)
        print("Art dataset shape {}".format(demo1_processed_phrase_extra.shape))

        demo1_processed_phrase = pd.concat(
            [demo1_processed_phrase_extra, pd.DataFrame(data_career)], ignore_index=True
        )
        print(
            "Final career and art sentences for Females {}".format(
                demo1_processed_phrase.shape
            )
        )

        demo1_processed_phrase.reset_index(inplace=True)

    drop_n = demo1_processed_phrase.shape[0] - 3000
    drop_indices = np.random.choice(demo1_processed_phrase.index, drop_n, replace=False)
    print(len(drop_indices))

    demo1_reduced = demo1_processed_phrase.drop(drop_indices)

    if topic_ins.name == "gender":
        demo1_reduced = demo1_reduced.drop(columns=["index", "id"])
    print(demo1_reduced.shape)
    demo1_reduced.to_csv(processed_anot_file_path, index=False)


def reddit_data_phrases_replace_target(topic_ins: Topic, bias_type: str = "bias"):
    """
    This script extracts Reddit phrases manually annotated as Biased and corresponding
    generates Counter target dataset
    """
    if not bias_type or bias_type not in ["bias", "bias_unbias"]:
        raise ValueError("Specify correct bias type")
    output_file_suffix = "_processed_phrase_biased"

    if bias_type == "bias_unbias":
        output_file_suffix = "_processed_phrase_biased_unbiased"

    suffix_file = "_processed_phrase_annotated"
    processed_anot_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix_file}.csv"
    )
    bias_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{output_file_suffix}.csv"
    )
    bias_maj_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{output_file_suffix}.csv"
    )
    demo1_df_processed = pd.read_csv(processed_anot_file_path, encoding="Latin-1")

    print("Shape of annotated dataframe {}".format(demo1_df_processed.shape))
    print(demo1_df_processed.head())

    if bias_type == "bias":
        demo1_df_processed = demo1_df_processed[demo1_df_processed["bias_phrase"] == 1]
    elif bias_type == "bias_unbias":
        demo1_df_processed = demo1_df_processed[
            (demo1_df_processed["bias_phrase"] == 1)
            | (demo1_df_processed["bias_phrase"] == 0)
        ]

    demo1_df_processed = demo1_df_processed.rename(
        columns={"phrase": "comments_processed"}
    )
    demo1_df_processed = demo1_df_processed.dropna(subset=["comments_processed"])

    print("Shape of biased dataframe {}".format(demo1_df_processed.shape))
    print(demo1_df_processed.head())

    demo1_df_processed.to_csv(bias_file_path, index=False)

    demo2_df = pd.DataFrame(
        columns=["initial_demo", "replaced_demo", "comments", "comments_processed"]
    )

    if topic_ins.name == "race":
        pairs = (
            ("black", "white"),
            ("african american", "anglo american"),
            ("african-american", "anglo-american"),
            ("afro-american", "anglo-american"),
            ("african", "american"),
            ("afroamericans", "angloamericans"),
            ("negroes", "caucasians"),
            ("dark-skin", "light-skin"),
            ("dark skin", "light skin"),
        )
    elif topic_ins.name == "religion1":
        pairs = (
            ("jew ", "christian "),
            ("jewish", "christian"),
            ("jews ", "christians "),
            ("judaism", "christianity"),
        )
    elif topic_ins.name == "religion2":
        pairs = (
            ("muslim", "christian"),
            ("islamic", "christian"),
            ("islam ", "christianity "),
            ("arabs", "americans"),
            ("islamism", "christianity"),
        )
    elif topic_ins.name == "gender":
        pairs = (
            ("woman", "man"),
            ("women", "men"),
            ("girl", "boy"),
            ("mother", "father"),
            ("daughter", "son"),
            ("wife", "husband"),
            ("niece", "nephew"),
            ("mom", "dad"),
            ("bride", "groom"),
            ("lady", "gentleman"),
            ("madam", "sir"),
            ("hostess", "host"),
            ("female", "male"),
            ("aunt", "uncle"),
            ("sister", "brother"),
            (" she ", " he "),
        )
    elif topic_ins.name == "orientation":
        pairs = (
            ("gay", "straight"),
            ("gays", "straight"),
            ("lesbian", "straight"),
            ("lesbians", "straight"),
            ("bisexual", "monosexual"),
            ("bisexuals", "monosexuals"),
            ("homosexual", "heterosexual"),
            ("homosexuals", "heterosexuals"),
            ("transgender", "cisgender"),
            ("transgenders", "cisgenders"),
            ("sapphic", "heterosexual"),
            ("pansexual", "heterosexual"),
            ("queer", "heterosexual"),
        )
    else:
        raise ValueError("Specify correct demographic")

    for idx, row in demo1_df_processed.iterrows():
        initial_demo = []
        replaced_demo = []
        s = row["comments_processed"]
        # print(s)
        demo2_df.at[idx, "comments"] = s

        for p in pairs:
            # s = s.replace(*p)
            if topic_ins.name == "race":
                if (
                    p[0] == "african"
                    and p[0] in s
                    and ("anglo american" in s or "anglo-american" in s)
                ):
                    s = s.replace(*p)
                elif (
                    p[1] == "american"
                    and p[1] in s
                    and ("anglo american" in s or "anglo-american" in s)
                ):
                    s = s.replace(*p)
                elif p[0] == "afro-american" and p[0] in s:
                    s = s.replace(*p)
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "religion1":
                if p[0] == "jewish":
                    if p[0] in s and ("christian" in s):
                        s = s.replace(*p)
                    elif "christian" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "religion2":
                if p[0] == "islamic":
                    if p[0] in s and ("christian" in s):
                        s = s.replace(*p)
                    elif "christian" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                elif p[0] == "islamism":
                    if p[0] in s and ("christianity" in s):
                        s = s.replace(*p)
                    elif "christianity" in s:
                        s = s.replace(*p)
                    else:
                        s = (
                            s.replace(p[0], "%temp%")
                            .replace(*reversed(p))
                            .replace("%temp%", p[1])
                        )
                else:
                    s = (
                        s.replace(p[0], "%temp%")
                        .replace(*reversed(p))
                        .replace("%temp%", p[1])
                    )
            elif topic_ins.name == "gender":
                s = s.replace(*p)
            elif topic_ins.name == "orientation":
                s = s.replace(*p)

            if p[1] in s and p[0] in row["comments_processed"]:
                initial_demo.append(p[0])
                replaced_demo.append(p[1])
        demo2_df.at[idx, "comments_processed"] = s
        demo2_df.at[idx, "initial_demo"] = initial_demo
        demo2_df.at[idx, "replaced_demo"] = replaced_demo

    print("Shape of demo2 data {}".format(demo2_df.shape))
    demo2_df.to_csv(bias_maj_file_path, index=False)


def reddit_data_text_train_test(
    topic_ins: Topic, group: Literal["min", "max"], bias_type: str = "bias"
):
    """
    This script generates csv and text files of train and test split for biased reddit dataset
    """
    pd.set_option("display.max_columns", 50)
    input_file_suffix = "_processed_phrase_biased"
    output_txt_train = "_bias_manual_train.txt"
    output_txt_test = "_bias_manual_valid.txt"
    output_csv_test = "_processed_phrase_biased_testset"
    output_csv_train = "_processed_phrase_biased_trainset"
    if bias_type == "bias_unbias":
        input_file_suffix = "_processed_phrase_biased_unbiased"
        output_txt_train = (
            "_bias_unbias_manual_train.txt"  # '_bias_manual_lowercase_train.txt'
        )
        output_txt_test = (
            "_bias_unbias_manual_valid.txt"  # '_bias_manual_lowercase_valid.txt'
        )
        output_csv_test = "_processed_phrase_biased_unbias_testset"
        output_csv_train = "_processed_phrase_biased_unbias_trainset"

    topic_group = (
        topic_ins.minority_group if group == "min" else topic_ins.majority_group
    )
    bias_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_group}{input_file_suffix}.csv"
    )
    df = pd.read_csv(bias_file_path)
    print("df shape {}".format(df.shape))

    if bias_type == "bias_unbias":
        suffix_file = "_processed_phrase_biased_testset_reduced"
        bias_unbias_file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_group}{suffix_file}.csv"
        )
        df_bias_testset = pd.read_csv(bias_unbias_file_path)
        cond = df["comments_processed"].isin(df_bias_testset["comments_processed"])
        df = df.drop(df[cond].index)

    print(df.shape)
    if topic_ins.name == "gender":
        train_test_ratio = 0.75
    else:
        train_test_ratio = 0.6

    df_train, df_test = train_test_split(
        df, stratify=df["bias_phrase"], train_size=train_test_ratio, random_state=1
    )

    print("Train {}".format(df_train.shape))
    print("Test {}".format(df_test.shape))
    print(df_train["bias_phrase"].value_counts())
    print(df_test["bias_phrase"].value_counts())

    desti_path_train = (
        files_path / topic_ins.name / f"{topic_ins.name}{output_txt_train}"
    )
    build_dataset_manual_annot(df_train, topic_ins.name, desti_path_train)

    desti_path_test = files_path / topic_ins.name / f"{topic_ins.name}{output_txt_test}"
    build_dataset_manual_annot(df_test, topic_ins.name, desti_path_test)
    output_csv_test_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_group}{output_csv_test}.csv"
    )
    df_test.to_csv(output_csv_test_path, index=False)
    output_csv_train_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_group}{output_csv_train}.csv"
    )
    df_train.to_csv(output_csv_train_path, index=False)


def reddit_data_valid_test_reduced(topic_ins: Topic):
    """
    Create test set and validation set split on the test dataset with removed perplexity outliers
    """
    pd.set_option("display.max_columns", 50)
    input_file_suffix = (
        "_processed_phrase_biased"  # '_processed_phrase_biased_unbiased'
    )

    output_csv_valid = (
        "_biased_valid_reduced"  # '_processed_phrase_biased_unbias_testset'
    )
    output_csv_test = (
        "_biased_test_reduced"  # '_processed_phrase_biased_unbias_trainset'
    )

    data_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{input_file_suffix}.csv"
    )
    df1 = pd.read_csv(data_file_path)
    data_file_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{input_file_suffix}.csv"
    )
    df2 = pd.read_csv(data_file_path)

    print("df1 shape {}".format(df1.shape))
    print("df2 shape {}".format(df2.shape))

    train_test_ratio = 0.5

    df1_valid, df1_test, df2_valid, df2_test = train_test_split(
        df1, df2, train_size=train_test_ratio, random_state=1
    )

    print("Train {}".format(df1_valid.shape))
    print("Test {}".format(df1_test.shape))
    print(df1_valid["comments_processed"].head())
    print(df1_test["comments_processed"].head())

    print("Train {}".format(df2_valid.shape))
    print("Test {}".format(df2_test.shape))
    print(df2_valid["comments_processed"].head())
    print(df2_test["comments_processed"].head())

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_{topic_ins.minority_group}{output_csv_valid}.txt"
    )
    build_dataset_manual_annot(df1_valid, topic_ins.name, desti_path, reduced=True)

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_{topic_ins.majority_group}{output_csv_valid}.txt"
    )
    build_dataset_manual_annot(df2_valid, topic_ins.name, desti_path, reduced=True)

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_{topic_ins.minority_group}{output_csv_test}.txt"
    )
    build_dataset_manual_annot(df1_test, topic_ins.name, desti_path, reduced=True)

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_{topic_ins.majority_group}{output_csv_test}.txt"
    )
    build_dataset_manual_annot(df2_test, topic_ins.name, desti_path, reduced=True)

    output_csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{output_csv_valid}.csv"
    )
    df1_valid.to_csv(output_csv_path, index=False)

    output_csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{output_csv_valid}.csv"
    )
    df2_valid.to_csv(output_csv_path, index=False)

    output_csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{output_csv_test}.csv"
    )
    df1_test.to_csv(output_csv_path, index=False)
    output_csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{output_csv_test}.csv"
    )
    df2_test.to_csv(output_csv_path, index=False)


def reddit_data_text_demo1_demo2(topic_ins: Topic):
    """
    This script generates text files of train datasets of Counter target data augmentation
    """
    suffix_file = "_processed_phrase_biased_trainset.csv"
    csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix_file}"
    )
    df_train_1 = pd.read_csv(csv_path)
    csv_path = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{suffix_file}"
    )
    df_train_2 = pd.read_csv(csv_path)

    df_train_1 = df_train_1[["comments_processed"]]
    df_train_2 = df_train_2[["comments_processed"]]

    df_train = pd.concat([df_train_1, df_train_2])

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_bias_manual_swapped_targets_train.txt"
    )
    build_dataset_manual_annot(df_train, desti_path)

    print(df_train.shape)


def reddit_data_phrases_replace_attribute(topic_ins: Topic, check_trainset: bool):
    """
    This script generates Counter attribute dataset for train and test set split
    """

    in_file_suffix = "_processed_phrase_biased_testset"
    if check_trainset:
        in_file_suffix = "_processed_phrase_biased_trainset"

    out_file_suffix = "_processed_phrase_unbiased_testset_pos_attr"
    if check_trainset:
        out_file_suffix = "_processed_phrase_unbiased_trainset_pos_attr"

    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{in_file_suffix}.csv"
    )
    demo1_df_processed = pd.read_csv(csv_file, encoding="Latin-1")

    print(demo1_df_processed.head())
    print(demo1_df_processed.shape)

    demo2_df = pd.DataFrame(
        columns=["initial_attr", "replaced_attr", "comments", "comments_processed"]
    )

    if topic_ins.name == "race":
        pairs = (
            ("violent", "peaceful"),
            ("abuse", "honour"),
            ("crash", "pure"),
            ("filth", "clean"),
            ("murder", "defend"),
            ("sickness", "health"),
            ("accident", "fortune"),
            ("death", "creation"),
            ("grief", "joy"),
            ("poison", "healthy"),
            ("stink", "fragrance"),
            ("assault", "protect"),
            ("disaster", "success"),
            ("hatred", "love"),
            ("pollute", "clean"),
            ("tragedy", "fortune"),
            ("divorce", "loyal"),
            ("jail", "university"),
            ("poverty", "wealth"),
            ("ugly", "beautiful"),
            ("kill", "save"),
            ("rotten", "brilliant"),
            ("vomit", "comfort"),
            ("agony", "relief"),
            ("prison", "university"),
            ("terrible", "awesome"),
            ("horrible", "pleasant"),
            ("nasty", "nice"),
            ("evil", "virtuous"),
            ("war", "peace"),
            ("awful", "charming"),
            ("failure", "success"),
            ("slave", "administrator"),
            ("criminal", "police"),
            ("homeless", "prosperous"),
            ("underclass", "privileged"),
            ("drug dealer", "police"),
            ("uncivilized", "civilized"),
            ("aggressive", "friendly"),
            ("oversexed", "moral"),
            ("crack addict", "disciplined"),
            ("animalistic", "noble"),
            ("dangerous", "innocent"),
            ("less intelligent", "intelligent"),
            ("predator", "protector"),
            ("unclean", "clean"),
            ("loud", "quiet"),
            ("drug addict", "police"),
            ("lazy", "active"),
            ("backward", "advanced"),
            ("bossy", "submissive"),
            ("headstrong", "obedient"),
            ("hyper sexual", "decent"),
        )
    elif topic_ins.name == "religion1":
        pairs = (
            ("greedy", "generous"),
            ("greed", "generosity"),
            ("meek ", "confident "),
            ("nerdy", "dummy"),
            ("nerd", "dummy"),
            ("circumci", "decircumci"),
            ("spoiled", "disciplined"),
            ("materialistic", "spiritual"),
            ("hooked", "short"),
            ("hook", "short"),
            ("beady", "blue"),
            ("large nose", "short nose"),
            ("big nose", "short nose"),
            ("red hair", "black hair"),
            ("red-hair", "black-hair"),
            ("lending", "borrowing"),
            ("miserly", "spendthrift"),
            ("miser", "spendthrift"),
            ("shady", "reputable"),
            ("frugal", "extravagant"),
            (" thrift ", "lavishness"),
            ("lust", "apathy"),
            ("overprotective", "careless"),
            ("loud", "quiet"),
            ("talkative", "untalkative"),
            ("overfeed", "underfeed"),
            ("aggressive", "peaceful"),
            ("pushy", "modest"),
            ("immigra", "emigra"),
            ("dominating", "submissive"),
            ("overbearing", "humble"),
            ("interfering", "unobtrusive"),
            ("selfish", "selfless"),
            ("devil", "angel"),
            ("evil", "good"),
            ("satanic", "godly"),
            ("satan", "god"),
            ("inferior", "superior"),
            ("horn", "no horn"),
            ("tail", "no tail"),
            ("manipulative", "innocent"),
            ("manipulating", "innocent"),
            ("cannibal", "civilized"),
            ("insatiable", "appeasable"),
            ("effeminate", "masculine"),
            ("conspiring", "neglecting"),
            ("conspire", "neglect"),
            ("conspiracies", "faithfulness"),
            ("conspiracy", "faithfulness"),
        )
    elif topic_ins.name == "religion2":
        pairs = (
            ("terrorist", "police"),
            ("terrorism", "nonviolence"),
            ("threaten", "defend"),
            ("threat", "safety"),
            ("dangerous", "helpful"),
            ("deadly", "safe"),
            ("attack", "defend"),
            ("bomb", "build"),
            ("murder", "restore"),
            ("hijack", "relinquish"),
            ("criminal", "lawful"),
            ("enemy", "friend"),
            ("war", "peace"),
            ("enemies", "friends"),
            ("extremist", "moderate"),
            ("radical", "conservative"),
            ("hate", "love"),
            ("rich", "poor"),
            ("lazy", "energetic"),
            ("flashy", "simple"),
            ("rapist", "defender"),
            ("slaves", "freemen"),
            ("shooter", "hero"),
            ("hijab", "hat"),
            ("vielded", "capped"),
            ("viel", "cap"),
            ("oppressed", "rejoiced"),
            ("patriar", "matriar"),
            ("misogyny", "philogyny"),
            ("deport", "admit"),
            ("detain", "liberate"),
            ("thieves", "police"),
            ("thief", "police"),
            ("thug", "hero"),
            ("charlatan", "honest"),
            ("brute", "mild"),
            ("power-hungry", "gratified"),
            ("beard", "clean-shave"),
            ("suicide bomber", "negotiator"),
            ("violence", "compassion"),
            ("violent", "gentle"),
            ("hostile", "kind"),
            ("sinister", "happy"),
        )
    elif topic_ins.name == "gender":
        pairs = (
            ("nurse", "surgeon"),
            ("secretary", "executive"),
            ("housekeeper", "manager"),
            ("cook", "officer"),
            ("sewing", "engineering"),
            ("cleaning", "programming"),
            ("waitress", "lawyer"),
            ("maid", "engineer"),
            ("child-care", "finance"),
            ("receptionist", "administrator"),
            ("clerk", "physician"),
            ("home-care", "science"),
            ("poetry", "math"),
            ("art", "science"),
            ("dance", "geometry"),
            ("literature", "technology"),
            ("novel", "equation"),
            ("symphony", "computation"),
            ("drama", "physics"),
            ("sculpture", "chemistry"),
            ("shakespeare", "einstein"),
        )
    elif topic_ins.name == "orientation":
        pairs = (
            ("mentally ill", "mentally strong"),
            ("polygamous", "monogamous"),
            ("flamboyant", "modest"),
            ("pedophile", "teleiophile"),
            ("sexual predator", "protector"),
            ("mental disorder", "mental stability"),
            ("villain", "hero"),
            ("serial killer", "police officer"),
            ("depressed", "cheerful"),
            ("depression", "cheerful"),
            ("artistic", "scientific"),
            ("drug addict", "healthy"),
            ("sadist", "compassionate"),
            ("theatre", "science"),
            ("dramatic", "subtle"),
            ("fashion", "unfashion"),
            ("rapist", "defender"),
            ("indecisive", "determined"),
            ("promiscuous", "moral"),
            ("prostitute", "chaste"),
            ("drag queen", "awesome"),
            ("criminal", "lawful"),
            ("hypersexual", "decent"),
            ("pervert", "saint"),
            ("faggot", "awesome"),
            ("weak", "strong"),
            ("timid", "daring"),
            ("loser", "winner"),
            ("lonely", "cheerful"),
            (" sin ", " virtue "),
            (" sin.", " virtue."),
            ("disease", "health"),
            ("confused", "assured"),
            ("strange", "normal"),
            ("weird", "normal"),
            ("coward", "brave"),
            ("immoral", "moral"),
            ("insecure", "confident"),
            ("repulsive", "delightful"),
            ("frustrated", "satisfied"),
            ("frustrating", "satisfying"),
            ("sinful", "innocent"),
            ("sensitive", "tough"),
            ("submissive", "dominating"),
            ("emotional", ("unemotional")),
        )
    else:
        raise ValueError("Specify correct demographic")

    for idx, row in demo1_df_processed.iterrows():
        initial_attr = []
        replaced_attr = []
        s = row["comments_processed"]
        # print(s)
        demo2_df.at[idx, "comments"] = s

        for p in pairs:
            s = s.replace(*p)

            if p[1] in s and p[0] in row["comments_processed"]:
                initial_attr.append(p[0])
                replaced_attr.append(p[1])

        demo2_df.at[idx, "comments_processed"] = s
        demo2_df.at[idx, "initial_attr"] = initial_attr
        demo2_df.at[idx, "replaced_attr"] = replaced_attr

    print("Shape of demo2 data {}".format(demo2_df.shape))
    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{out_file_suffix}.csv"
    )
    demo2_df.to_csv(csv_file, index=False)


def reddit_data_text_bias_unbias(topic_ins: Topic):
    """
    This script generates text files of train and test datasets of Counter
    attribute data augmentation
    """
    suffix = "_processed_phrase_biased_trainset"
    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix}.csv"
    )
    df_train_1 = pd.read_csv(csv_file)

    suffix = "_processed_phrase_unbiased_trainset_pos_attr"
    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix}.csv"
    )
    df_train_2 = pd.read_csv(csv_file)

    df_train_1 = df_train_1[["comments_processed"]]
    df_train_2 = df_train_2[["comments_processed"]]

    df_train = pd.concat([df_train_1, df_train_2])

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_bias_manual_swapped_attr_train.txt"
    )
    build_dataset_manual_annot(df_train, None, desti_path)

    print(df_train.shape)

    suffix = "_processed_phrase_biased_testset"
    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix}.csv"
    )
    df_test_1 = pd.read_csv(csv_file)

    suffix = "_processed_phrase_unbiased_testset_pos_attr"
    csv_file = (
        data_path
        / topic_ins.name
        / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{suffix}.csv"
    )
    df_test_2 = pd.read_csv(csv_file)

    df_test_1 = df_test_1[["comments_processed"]]
    df_test_2 = df_test_2[["comments_processed"]]

    df_test = pd.concat([df_test_1, df_test_2])

    print(df_test.shape)

    desti_path = (
        files_path
        / topic_ins.name
        / f"{topic_ins.name}_bias_manual_swapped_attr_test.txt"
    )
    build_dataset_manual_annot(df_test, None, desti_path)
