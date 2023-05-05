from textattack.goal_functions.classification import ClassificationGoalFunction
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance, BERTScore
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations.word_swaps import WordSwapEmbedding
import textattack
import transformers

import torch

import csv
import os


class AdaptiveClassificationAttack(ClassificationGoalFunction):
    def __init__(
        self,
        *args,
        target_confidence_score=0.5,
        defense=None,
        detection_target_score=None,
        log_file=None,
        **kwargs,
    ):
        self.log_file = log_file
        # Create log file with header if it doesn't exist
        if self.log_file is not None and not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if defense is not None:
                    writer.writerow(
                        [
                            "attacked_text",
                            "attack_model_output",
                            "defense_score",
                            "num_queries",
                            "improvement",
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            "attacked_text",
                            "attack_model_output",
                            "num_queries",
                            "improvement",
                        ]
                    )

        self.target_confidence_score = target_confidence_score
        self.defense = defense
        self.detection_target_score = detection_target_score
        self.best_result = (0, 0)
        self.info_to_write = []
        # self.num_queries = 0
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, attacked_text):
        if (1 - model_output[self.ground_truth_output]) >= self.target_confidence_score:
            if self.defense is not None:
                defense_score = 1 - self.defense(attacked_text)
                return defense_score <= self.detection_target_score
            else:
                return True
        else:
            return False

    def _get_score(self, model_output, attacked_text):
        # self.num_queries += 1
        attack_score = 1 - model_output[self.ground_truth_output]

        if self.defense is not None:
            defense_score = 1 - self.defense(attacked_text)
            final_score = (
                min(attack_score, self.target_confidence_score) + defense_score
            )
        else:
            final_score = attack_score

        improvement = False
        if len(self.info_to_write) >= 1000:
            if self.log_file is not None:
                with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.info_to_write)
                self.info_to_write = []

        # Update the best result
        if final_score > self.best_result[0] or self.num_queries < self.best_result[1]:
            improvement = True
            self.best_result = (final_score, self.num_queries)

        # Log the result to the log_file
        if self.log_file is not None:
            info = [
                attacked_text.text,
                attack_score,
                self.num_queries,
                improvement,
            ]
            # print(self.num_queries, info)
            if self.defense is not None:
                defense_score = 1 - self.defense(attacked_text)
                info.insert(2, defense_score)

            self.info_to_write.append(info)

        # Update the best result

        return final_score


def load_model_and_tokenizer(dataset="imdb"):
    model_name = f"textattack/distilbert-base-uncased-{dataset}"
    # model_name = "textattack/bert-base-uncased-imdb"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

    # model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     "textattack/distilbert-base-uncased-imdb"
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "textattack/distilbert-base-uncased-imdb"
    # )


def get_attack(dataset="imdb", log_file=None):
    model, tokenizer = load_model_and_tokenizer(dataset)

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # Construct our four components for `Attack`

    goal_function = AdaptiveClassificationAttack(
        model_wrapper,
        target_confidence_score=0.999,
        log_file=log_file,
    )

    constraints = [
        RepeatModification(),
        StopwordModification(),
        WordEmbeddingDistance(min_cos_sim=0.8),
        UniversalSentenceEncoder(),
        # BERTScore(min_bert_score=0.9),
    ]
    transformations = textattack.transformations.CompositeTransformation(
        [
            WordSwapEmbedding(),
        ]
    )
    search_method = GreedyWordSwapWIR(wir_method="gradient")

    attack = textattack.Attack(
        goal_function, constraints, transformations, search_method
    )
    attack.cuda_()
    return attack


def change_attack_goal(attack, goal):
    attack = textattack.Attack(
        goal_function=goal,
        constraints=attack.constraints,
        transformation=attack.transformation,
        search_method=attack.search_method,
    )
    return attack


if __name__ == "__main__":
    attack = get_attack(dataset="ag-news", log_file="all_log_yelp.csv")
    dataset = textattack.datasets.HuggingFaceDataset("ag_news", split="test")
    attack_args = textattack.AttackArgs(
        num_examples=-1,
        # log_to_csv="good_very_very_far_boundary.csv",
        # checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        # disable_stdout=False,
        enable_advance_metrics=True,
        num_examples_offset=132,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    print(results)
    # original_text = "This movie is great!"
    # label = 1
    # attacked_text = attack.attack(
    #     "I really enjoyed the new movie that came out last month.", label
    # )
    # print(attacked_text)
