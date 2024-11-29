import pandas as pd
import numpy as np
from scipy import stats
from app.schemas.redditbias import Topic
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import time
import logging
from app.core.config import (
    redditbias_data_path as data_path,
    redditbias_execution_logs_path as exp_path,
)
from app.utils.redditbias_original_functions import find_anomalies, get_perplexity_list


def redditbias_measure_bias(
    topic_ins: Topic,
    on_set: bool = True,
    get_perplexity: bool = True,
    reduce_set: bool = False,
):
    """
    This script performs Student t-test on the perplexity distribution of two sentences groups
    with contrasting targets
    """
    start = time.time()
    input_file_suffix = (
        "_biased_test_reduced"  # '_processed_phrase_biased_testset_reduced' #
    )
    output_file_suffix = "_perplex_phrase_biased"  # '_perplex'

    pretrained_model = "microsoft/DialoGPT-small"  # 'bert_base_uncased' # 'gpt2'

    on_set_path = "" if on_set else "_test"
    logging.basicConfig(
        filename=(exp_path / f"measure_bias_{topic_ins.name}{on_set_path}.log"),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
    )

    pd.set_option("max_colwidth", 600)
    pd.options.display.max_columns = 10

    if get_perplexity:
        print("Calculating perplexity for topic_ins.name: {}".format(topic_ins.name))
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{input_file_suffix}.csv"
        )
        race_df = pd.read_csv(file_path)
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{input_file_suffix}.csv"
        )
        race_df_2 = pd.read_csv(file_path)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
        print(
            "Done with demo1 perplexity in {} on set".format((time.time() - start) / 60)
        )
        race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

        logging.info(
            "Time to get perplexity scores {}".format((time.time() - start) / 60)
        )
        race_df["perplexity"] = race_1_perplexity
        race_df_2["perplexity"] = race_2_perplexity
    else:
        logging.info("Getting saved perplexity")
        print("Getting saved perplexity")
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{output_file_suffix}.csv"
        )
        race_df = pd.read_csv(file_path)
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{output_file_suffix}.csv"
        )
        race_df_2 = pd.read_csv(file_path)
        race_1_perplexity = race_df["perplexity"]
        race_2_perplexity = race_df_2["perplexity"]

    logging.debug(
        "Instances in {} 1 and 2: {}, {}".format(
            topic_ins.name, len(race_1_perplexity), len(race_2_perplexity)
        )
    )
    logging.debug(
        "Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}".format(
            np.mean(race_1_perplexity), np.var(race_1_perplexity)
        )
    )
    logging.debug(
        "Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}".format(
            np.mean(race_2_perplexity), np.var(race_2_perplexity)
        )
    )

    print(
        "Mean and std of unfiltered perplexities demo1 - Mean {}, Std {}".format(
            np.mean(race_1_perplexity), np.std(race_1_perplexity)
        )
    )
    print(
        "Mean and std of unfiltered perplexities demo2 - Mean {}, Std {}".format(
            np.mean(race_2_perplexity), np.std(race_2_perplexity)
        )
    )

    print(len(race_1_perplexity), len(race_2_perplexity))

    demo1_out = find_anomalies(np.array(race_1_perplexity))
    demo2_out = find_anomalies(np.array(race_2_perplexity))

    print(demo1_out, demo2_out)

    for i, (p1, p2) in enumerate(zip(race_1_perplexity, race_2_perplexity)):
        if p1 in demo1_out or p2 in demo2_out:
            print(
                "Outlier in demo1 is {}".format(
                    race_df.loc[race_df["perplexity"] == p1]
                )
            )
            print(
                "Outlier in demo2 is {}".format(
                    race_df_2.loc[race_df_2["perplexity"] == p2]
                )
            )
            race_df.drop(race_df.loc[race_df["perplexity"] == p1].index, inplace=True)
            race_df_2.drop(
                race_df_2.loc[race_df_2["perplexity"] == p2].index, inplace=True
            )

    if reduce_set:
        print("DF shape after reducing {}".format(race_df.shape))
        print("DF 2 shape after reducing {}".format(race_df_2.shape))
        file_args = f"{topic_ins.name}_{topic_ins.minority_group}{input_file_suffix}"
        output_file_path = (
            data_path / topic_ins.name / f"reddit_comments_{file_args}_reduced.csv"
        )
        race_df.to_csv(output_file_path, index=False)

        file_args = f"{topic_ins.name}_{topic_ins.majority_group}{input_file_suffix}"
        output_file_path = (
            data_path / topic_ins.name / f"reddit_comments_{file_args}_reduced.csv"
        )
        race_df_2.to_csv(output_file_path, index=False)

        print(len(race_df["perplexity"]), len(race_df_2["perplexity"]))
        print(
            "Mean and std of filtered perplexities demo1 - Mean {}, Std {}".format(
                np.mean(race_df["perplexity"]), np.std(race_df["perplexity"])
            )
        )
        print(
            "Mean and std of filtered perplexities demo2 - Mean {}, Std {}".format(
                np.mean(race_df_2["perplexity"]), np.std(race_df_2["perplexity"])
            )
        )

        t_unpaired, p_unpaired = stats.ttest_ind(
            race_df["perplexity"].to_list(),
            race_df_2["perplexity"].to_list(),
            equal_var=False,
        )
        print(
            "Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}".format(
                t_unpaired, p_unpaired
            )
        )

        t_paired, p_paired = stats.ttest_rel(
            race_df["perplexity"].to_list(), race_df_2["perplexity"].to_list()
        )
        print(
            "Paired t-test, after outlier removal: t-value {}, p-value {}".format(
                t_paired, p_paired
            )
        )

    t_value, p_value = stats.ttest_rel(race_1_perplexity, race_2_perplexity)

    print(
        "Mean and std of unfiltered perplexities demo1 - Mean {}, Std {}".format(
            np.mean(race_1_perplexity), np.std(race_1_perplexity)
        )
    )
    print(
        "Mean and std of unfiltered perplexities demo2 - Mean {}, Std {}".format(
            np.mean(race_2_perplexity), np.std(race_2_perplexity)
        )
    )
    print(
        "Unfiltered perplexities - T value {} and P value {}".format(t_value, p_value)
    )
    print(t_value, p_value)

    logging.info("Total time taken {}".format((time.time() - start) / 60))


def reddit_measure_bias_attribute_swap(
    topic_ins: Topic,
    on_set: bool = True,
    get_perplexity: bool = True,
    reduce_set: bool = False,
):
    """
    This script performs Student t-test on the perplexity distribution of two
    sentences groups with contrasting attributes
    """
    start = time.time()

    input_file_biased = "_processed_phrase_biased_testset.csv"
    # '_processed_phrase_biased' # '_processed_phrase_biased_testset'
    # '_processed_sent_biased' # '_processed'
    input_file_unbiased = "_processed_phrase_unbiased_testset_pos_attr.csv"
    out_file_suffix = "_perplex_phrase_attribute_swap_biased.csv"

    pretrained_model = "microsoft/DialoGPT-small"
    # 'gpt2' # 'roberta-base' # 'bert-base-uncased' #'ctrl'
    # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'minimaxir/reddit' # 'xlnet-large-cased'

    on_set_path = "" if on_set else "_test"
    logging.basicConfig(
        filename=(
            exp_path / f"measure_bias_attr_swap_{topic_ins.name}{on_set_path}.log"
        ),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
    )

    pd.set_option("max_colwidth", 600)
    pd.options.display.max_columns = 10

    if get_perplexity:
        logging.info("Calculating perplexity")
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{input_file_biased}"
        )
        race_df = pd.read_csv(file_path)
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{input_file_unbiased}"
        )
        race_df_2 = pd.read_csv(file_path)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        # model = AutoModelWithLMAndDebiasHead.from_pretrained(pretrained_model,
        # debiasing_head=debiasing_head)
        # model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
        print(
            "Done with demo1 perplexity in {} on set".format((time.time() - start) / 60)
        )
        race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

        # model_perp = get_model_perplexity(race_df, model, tokenizer)
        # print('Model perplexity {}'.format(model_perp))

        logging.info(
            "Time to get perplexity scores {}".format((time.time() - start) / 60)
        )
        race_df["perplexity"] = race_1_perplexity
        race_df_2["perplexity"] = race_2_perplexity

        # race_df.to_csv(data_path + topic_ins.name + '/' + 'reddit_comments_' + topic_ins.name
        # + '_' + topic_ins.minority_group + output_file_suffix + '.csv')
        # race_df_2.to_csv(data_path + topic_ins.name + '/' + 'reddit_comments_' + topic_ins.name
        # + '_' + topic_ins.majority_group + output_file_suffix +'.csv')
    else:
        logging.info("Getting saved perplexity")
        print("Getting saved perplexity")
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{out_file_suffix}"
        )
        race_df = pd.read_csv(file_path)
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.majority_group}{out_file_suffix}"
        )
        race_df_2 = pd.read_csv(file_path)
        race_1_perplexity = race_df["perplexity"]
        race_2_perplexity = race_df_2["perplexity"]

    print(
        "Instances in topic_ins.name 1 and 2: {}, {}".format(
            len(race_1_perplexity), len(race_2_perplexity)
        )
    )
    print(
        "Mean and Std of unfiltered perplexities demo1 - Mean {}, Variance {}".format(
            np.mean(race_1_perplexity), np.std(race_1_perplexity)
        )
    )
    print(
        "Mean and Std of unfiltered perplexities demo2 - Mean {}, Variance {}".format(
            np.mean(race_2_perplexity), np.std(race_2_perplexity)
        )
    )

    assert len(race_1_perplexity) == len(race_2_perplexity)

    demo1_out = find_anomalies(np.array(race_1_perplexity))
    demo2_out = find_anomalies(np.array(race_2_perplexity))

    print(demo1_out, demo2_out)

    for i, (p1, p2) in enumerate(zip(race_1_perplexity, race_2_perplexity)):
        if p1 in demo1_out or p2 in demo2_out:
            race_df.drop(race_df.loc[race_df["perplexity"] == p1].index, inplace=True)
            race_df_2.drop(
                race_df_2.loc[race_df_2["perplexity"] == p2].index, inplace=True
            )

    if reduce_set:
        print("DF shape after reducing {}".format(race_df.shape))
        print("DF 2 shape after reducing {}".format(race_df_2.shape))
        reduced_bias = input_file_biased + "_neg_attr_reduced"
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{reduced_bias}.csv"
        )
        race_df.to_csv(file_path, index=False)
        reduced_unbias = input_file_unbiased + "_reduced"
        file_path = (
            data_path
            / topic_ins.name
            / f"reddit_comments_{topic_ins.name}_{topic_ins.minority_group}{reduced_unbias}.csv"
        )
        race_df_2.to_csv(file_path, index=False)

    print(
        "Mean and Std of filtered perplexities demo1 - Mean {}, Variance {}".format(
            np.mean(race_df["perplexity"]), np.std(race_df["perplexity"])
        )
    )
    print(
        "Mean and Std of filtered perplexities demo2 - Mean {}, Variance {}".format(
            np.mean(race_df_2["perplexity"]), np.std(race_df_2["perplexity"])
        )
    )

    t_value, p_value = stats.ttest_ind(
        race_1_perplexity, race_2_perplexity, equal_var=False
    )

    print(
        "Unfiltered perplexities - T value {} and P value {}".format(t_value, p_value)
    )
    print(t_value, p_value)

    print(len(race_df["perplexity"]), len(race_df_2["perplexity"]))

    t_unpaired, p_unpaired = stats.ttest_ind(
        race_df["perplexity"].to_list(),
        race_df_2["perplexity"].to_list(),
        equal_var=False,
    )
    print(
        "Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}".format(
            t_unpaired, p_unpaired
        )
    )

    t_paired, p_paired = stats.ttest_rel(
        race_df["perplexity"].to_list(), race_df_2["perplexity"].to_list()
    )
    print(
        "Paired t-test, after outlier removal: t-value {}, p-value {}".format(
            t_paired, p_paired
        )
    )

    logging.info("Total time taken {}".format((time.time() - start) / 60))
