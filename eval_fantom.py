import os
import json
import argparse
import random
import evaluate
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

import task.dataset_loader as loader
from agents.gpt import GPT3BaseAgent, ConversationalGPTBaseAgent
from agents.huggingface import FlanT5Agent, FlanUL2Agent, MistralAIAgent, ZephyrAgent
from agents.together_ai import TogetherAIAgent

PROJECT_HOME = Path(__file__).parent.resolve()
DATA_DIR = 'data'
DATA_DIR_PATH = os.path.join(PROJECT_HOME, DATA_DIR)
EVAL_DIR_PATH = os.path.join(DATA_DIR_PATH, 'results')
RANDOM_SEED = 99
random.seed(RANDOM_SEED)

class FantomDataset(Dataset):
    def __init__(self, texts, args):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        return text

class FantomEvalAgent():
    def __init__(self, args):
        self.args = args
        self.prompt_header = "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, based on the following in-person conversation between individuals who have just met.\n\n"
        self.output_filename_suffix = '_{}_input_{}_cot-{}.json'.format(self.args.conversation_input_type, self.args.model, self.args.use_cot)
        self.load_fantom()
        self.setup_fantom()

        self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)

    def load_fantom(self):
        self.fantom_df = loader.load()

    def respond(self, prompt):
        response = self.model.interact(prompt)
        return response

    def load_model(self):
        if self.args.model.startswith("text-"):
            model = GPT3BaseAgent({'engine': self.args.model, 'temperature': 0, 'top_p': 0.95, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
        elif self.args.model.startswith("gpt-"):
            model = ConversationalGPTBaseAgent({'model': self.args.model, 'temperature': 0, 'top_p': 0.95, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
        elif self.args.model.startswith('flan-t5'):
            model = FlanT5Agent(self.args)
        elif self.args.model.startswith('flan-ul2'):
            model = FlanUL2Agent(self.args)
        elif self.args.model.endswith('-tg'):
            model = TogetherAIAgent(self.args.__dict__)
        elif self.args.model.startswith('mistral'):
            model = MistralAIAgent(self.args)
        elif self.args.model.startswith('zephyr'):
            model = ZephyrAgent(self.args)
        else:
            raise NotImplementedError

        return model

    def compute_f1(self, ground_truth, model_response):
        """
        Compute the F1 score between the ground truth and model response.

        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.

        Returns:
            float: The F1 score.
        """
        ground_truth = ground_truth.split()
        model_response = model_response.split()
        common = Counter(ground_truth) & Counter(model_response)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(model_response)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate_belief_q(self, qa, model_response, metric='cosine'):
        """
        Evaluate the belief question by comparing the model's response with the correct answer and wrong answer.

        Args:
            qa (dict): A dictionary containing the question and answers.
            model_response (str): The model's response to the question.
            metric (str, optional): The similarity metric to use for comparison. Defaults to 'cosine'.

        Returns:
            tuple: A tuple containing a boolean value indicating if the model's response matches the correct answer,
                   and the lexical overlap score between the model's response and the corresponding answer.
        """
        wrong_tom_view = qa['wrong_answer']
        if metric == "cosine":
            wrong_tom_view_emb = self.embedder.encode(wrong_tom_view)
            personx_view_emb = self.embedder.encode(qa['correct_answer'])
            model_response_emb = self.embedder.encode(model_response)
            similarity_wrong_tom_view = cosine_similarity(model_response_emb.reshape(1, -1), wrong_tom_view_emb.reshape(1, -1))[0][0]
            similarity_personx_view = cosine_similarity(model_response_emb.reshape(1, -1), personx_view_emb.reshape(1, -1))[0][0]
        else:
            raise NotImplementedError

        if similarity_wrong_tom_view >= similarity_personx_view:
            wrong_view_lexical_overlap = self.compute_f1(wrong_tom_view, model_response)
            return False, wrong_view_lexical_overlap
        else:
            personx_view_lexical_overlap = self.compute_f1(qa['correct_answer'], model_response)
            return True, personx_view_lexical_overlap

    def evaluate_mc_belief_q(self, qa, model_response):
        """
        Evaluate the multiple-choice version belief question.

        Args:
            qa (dict): The question and answer information.
            model_response (str): The model's response to the question.

        Returns:
            bool: True if the model's response matches the correct answer, False otherwise.
        """
        int_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        answer = int_to_alphabet[qa['correct_answer']]
        response = model_response.lower()

        if response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + ".") or response.startswith(answer + ":") or response.startswith(answer + ",") or "({})".format(answer) in response or answer == response: # a) or a. or a or (a)
            return True
        else:
            return False

    def evaluate_list_q(self, qa, model_response):
        """
        Check whether all the characters in the correct answer are in the model's response
        and none of the characters in the wrong answer are in the model's response

        Args:
            qa (dict): A dictionary containing the question and answer information.
            model_response (str): The response generated by the model.

        Returns:
            tuple: A tuple containing three values:
                - A boolean indicating whether the model's response satisfies the evaluation criteria.
                - A boolean indicating whether any aware characters were excluded from the model's response.
                - A boolean indicating whether any unaware characters were included in the model's response.
        """
        excluded_aware_character = False
        included_unaware_character = False
        for character in qa['correct_answer']:
            if character.lower() not in model_response.lower():
                excluded_aware_character = True
                break

        for character in qa['wrong_answer']:
            if character.lower() in model_response.lower():
                included_unaware_character = True
                break

        return not(excluded_aware_character or included_unaware_character), excluded_aware_character, included_unaware_character

    def map_binary_answer_to_int(self, model_response):
        """
        Maps a binary answer to an integer value.

        Args:
            model_response (str): The model's response.

        Returns:
            int: The mapped integer value. Returns 1 for positive answers (e.g., 'yes', 'true'), 
                 0 for negative answers (e.g., 'no', 'false'), and -1 for other cases.
        """
        model_answer = model_response.lower().strip("'").strip('"')
        if " yes," in model_answer or " yes " in model_answer or model_answer.startswith("yes") or " yes." in model_answer or " knows " in model_answer or model_answer.lower().startswith("true"):
            return 1
        elif " no," in model_answer or " no " in model_answer or model_answer.startswith("no") or " no." in model_answer or " does not know " in model_answer or " doesn't know " in model_answer or model_answer.lower().startswith("false"):
            return 0
        else:
            return -1

    def evaluate_binary_q_with_f1(self, qa, model_response):
        """
        Evaluates a binary question with F1 score.

        Args:
            qa (dict): A dictionary containing the question and correct answer.
            model_response (str): The response generated by the model.

        Returns:
            bool: True if the model's response contains the correct answer, False otherwise.
        """
        tom_answer = qa['correct_answer'].split(":")[0] # for no:long
        model_answer = model_response.split()[0].lower().strip(",")
        if tom_answer in model_answer:
            return True
        else:
            return False

    def evaluate_fact_q(self, qa, model_response):
        result = self.compute_f1(qa['correct_answer'].lower(), model_response.lower())
        return result

    def yesno_to_int(self, yesno_str):
        mapping = {'yes': 1, 'no': 0, 'no:long': 0, 'error': -1}
        return mapping[yesno_str]

    def evaluate_response(self, qas, predictions):
        """
        Evaluates the model's response for a list of questions and predictions.

        Args:
            qas (list): List of question-answer pairs.
            predictions (list): List of model predictions.

        Returns:
            list: Updated list of question-answer pairs with evaluation results and predictions.
        """
        print("Running evaluation...")
        assert len(qas) == len(predictions), "Number of questions and model predictions should be the same."

        for qa, pred in tqdm(zip(qas, predictions), total=len(qas)):
            if qa['question_type'].startswith("tom:belief:"):
                if qa['question_type'].endswith(":multiple-choice"):
                    result = self.evaluate_mc_belief_q(qa, pred)
                else:
                    result, word_overlap = self.evaluate_belief_q(qa, pred)
                    qa['word_overlap'] = word_overlap
            elif qa['question_type'].endswith(":list"):
                result, excluded_aware_character, included_unaware_character = self.evaluate_list_q(qa, pred)
                qa['excluded_aware_character'] = excluded_aware_character
                qa['included_unaware_character'] = included_unaware_character
            elif qa['question_type'].endswith(":binary"):
                _binary_answer = self.map_binary_answer_to_int(pred)
                if self.yesno_to_int(qa['correct_answer']) == _binary_answer:
                    result = True
                else:
                    result = False
                qa['binarized_model_answer'] = _binary_answer
            elif qa['question_type'].startswith("fact"):
                result = self.evaluate_fact_q(qa, pred)
            else:
                raise NotImplementedError

            qa['result'] = result
            qa['prediction'] = pred

        return qas

    def score_and_analyze(self, df, target_scenario='inaccessible'):
        """
        Aggregates scores and performs analysis on the model responses and evaluation results.

        Args:
            df (pandas.DataFrame): The dataframe containing the FANToM QA pairs, model responses, and evaluation results.
            target_scenario (str, optional): The target scenario for analysis. Defaults to 'inaccessible'.

        Returns:
            dict: A dictionary containing the calculated scores and analysis results.
        """
        report = {'model': self.args.model, 'conversation_input_type': self.args.conversation_input_type}
        f1_metric = evaluate.load("f1")
        aggregation_target = self.args.aggregation_target + "_id"
        tom_df = df[df['question_type'].str.startswith("tom")]
        target_df = tom_df[tom_df['missed_info_accessibility'] == target_scenario].copy()

        ############# Scores #############
        # ALL* score
        report[target_scenario+':set:ALL*'] = target_df.groupby(aggregation_target)['result'].all().mean()

        # ALL score
        target_question_for_all = ["tom:belief:"+target_scenario+":multiple-choice", "tom:answerability:list", "tom:answerability:binary", "tom:info_accessibility:list", "tom:info_accessibility:binary"]
        report[target_scenario+':set:ALL'] = target_df[target_df['question_type'].isin(target_question_for_all)].groupby(aggregation_target)['result'].all().mean()

        # Belief Questions: multiple-choice, dist., f1
        report[target_scenario+':belief:multiple-choice'] = target_df[target_df['question_type'].str.endswith(":multiple-choice")]['result'].mean()
        report[target_scenario+':belief:distance'] = target_df[target_df['question_type'] == "tom:belief:"+target_scenario]['result'].mean()
        report[target_scenario+':belief_true_word-f1'] = target_df[(target_df['question_type'] == "tom:belief:"+target_scenario) & (target_df['result'] == True)]['word_overlap'].mean()

        # Answerability Questions: ALL, list, binary
        report[target_scenario+':answerability:set:ALL'] = target_df[target_df['question_type'].str.startswith("tom:answerability")].groupby(aggregation_target)['result'].all().mean()
        report[target_scenario+':answerability:list'] = target_df[target_df['question_type'] == "tom:answerability:list"]['result'].mean()
        answerability_model_responses = target_df[target_df['question_type'] == 'tom:answerability:binary']['binarized_model_answer'].to_list()
        answerability_references = target_df[target_df['question_type'] == 'tom:answerability:binary']['correct_answer'].map(self.yesno_to_int).to_list()
        report[target_scenario+':answerability:binary-f1'] = f1_metric.compute(predictions=answerability_model_responses, references=answerability_references, pos_label=0, average="weighted")['f1']

        # Info Accessibility Questions: All, list, binary
        report[target_scenario+':info_accessibility:set:ALL'] = target_df[target_df['question_type'].str.startswith("tom:info_accessibility")].groupby(aggregation_target)['result'].all().mean()
        report[target_scenario+':info_accessibility:list'] = target_df[target_df['question_type']=="tom:info_accessibility:list"]['result'].mean()
        accessibility_model_responses = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['binarized_model_answer'].to_list()
        accessibility_references = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['correct_answer'].map(self.yesno_to_int).to_list()
        report[target_scenario+':info_accessibility:binary-f1'] = f1_metric.compute(predictions=accessibility_model_responses, references=accessibility_references, pos_label=0, average="weighted")['f1']

        # Fact Questions
        report['fact_word-f1'] = df[df['question_type'].str.startswith("fact")]['result'].mean()


        ############# Error Analysis #############
        # why the model got the list questions wrong
        list_wrong = target_df[(target_df['question_type']=="tom:answerability:list") & (target_df['result'] == False)][['excluded_aware_character', 'included_unaware_character']].copy()
        list_wrong['both'] = list_wrong['excluded_aware_character'] & list_wrong['included_unaware_character']
        list_wrong['reason'] = list_wrong.apply(lambda x: 'did_both' if x['both'] else 'excluded_aware_character' if x['excluded_aware_character'] else 'included_unaware_character', axis=1)
        report[target_scenario+':tom:lists:wrong_reasons:freq'] = list_wrong['reason'].value_counts(normalize=False).to_dict()

        # why the model got the binary questions wrong
        binary_wrong_reasons = target_df[(target_df['question_type'].str.endswith(":binary")) & (target_df['result'] == False)]['binarized_model_answer'].value_counts(normalize=False).to_dict()
        if 0 in binary_wrong_reasons.keys():
            binary_wrong_reasons['false_negative'] = binary_wrong_reasons.pop(0)
        if 1 in binary_wrong_reasons.keys():
            binary_wrong_reasons['false_positive'] = binary_wrong_reasons.pop(1)
        if -1 in binary_wrong_reasons.keys():
            binary_wrong_reasons['irrelevant_response'] = binary_wrong_reasons.pop(-1)
        report[target_scenario+':tom:binary:wrong_reasons:freq'] = binary_wrong_reasons


        ############# More Analysis #############
        # 1. Results for each tom_order type in Belief questions: first order and second order
        belief_df = tom_df[tom_df['question_type'] == ('tom:belief:' + target_scenario)].copy() # XXX: only consider the BeliefQ[dist.] questions
        belief_df['tom_order'] = belief_df['tom_type'].map(lambda x: x.split(":")[0])
        tom_order_results = belief_df.groupby('tom_order')['result'].value_counts(normalize=True)
        for idx in tom_order_results.index:
            if idx[1] == True:
                report[target_scenario + ":" + idx[0]] = tom_order_results[idx]

        # 2. Cyclic vs Acyclic second order belief questions
        belief_results = belief_df.groupby('tom_type')['result'].value_counts(normalize=True)
        for idx in belief_results.index:
            if idx[1] == True:
                report[target_scenario + ":" + idx[0]] = belief_results[idx]

        # 3. Character tracking analysis 
        binary_qas = target_df[(target_df['question_type'].str.endswith(":binary"))].copy()
        binary_qas['target_character'] = binary_qas['question'].map(lambda x: x.removeprefix("Does ").split(" know")[0].lower())
        belief_qas = target_df[(target_df['question_type'].str.startswith("tom:belief"))].copy()
        belief_qas['target_character'] = belief_qas['question'].map(lambda x: x.lower().split("does ")[1].split()[0].lower())
        answerability_list_qas = target_df[target_df['question_type'].str.endswith("answerability:list")].set_index(aggregation_target, drop=False)
        accessibility_list_qas = target_df[target_df['question_type'].str.endswith("info_accessibility:list")].set_index(aggregation_target, drop=False)

        # Tile the list question responses to the binary question level for each character
        binary_answerability_qas = binary_qas[binary_qas['question_type'].str.startswith('tom:answerability:')]
        tiled_answerability_list_qas = binary_answerability_qas[[aggregation_target, 'target_character', 'correct_answer']].join(answerability_list_qas[['prediction', aggregation_target]], on=aggregation_target, how='outer', lsuffix='-binary')
        tiled_answerability_list_qas['binarized_model_answer'] = tiled_answerability_list_qas.apply(lambda x: x['target_character'].lower() in x['prediction'].lower(), axis=1)
        tiled_answerability_list_qas['binarized_correct_answer'] = tiled_answerability_list_qas['correct_answer'].map(lambda x: True if x =='yes' else False)
        tiled_answerability_list_qas['result'] = tiled_answerability_list_qas.apply(lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], axis=1)

        binary_accessibility_qas = binary_qas[binary_qas['question_type'].str.startswith('tom:info_accessibility:')]
        tiled_accessibility_list_qas = binary_accessibility_qas[[aggregation_target, 'target_character', 'correct_answer']].join(accessibility_list_qas[['prediction', aggregation_target]], on=aggregation_target, how='outer', lsuffix='-binary')
        tiled_accessibility_list_qas['binarized_model_answer'] = tiled_accessibility_list_qas.apply(lambda x: x['target_character'].lower() in x['prediction'].lower(), axis=1)
        tiled_accessibility_list_qas['binarized_correct_answer'] = tiled_accessibility_list_qas['correct_answer'].map(lambda x: True if x =='yes' else False)
        tiled_accessibility_list_qas['result'] = tiled_accessibility_list_qas.apply(lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], axis=1)

        df_for_all_character_metric = pd.concat([binary_qas[['target_character', aggregation_target, 'result']], belief_qas[['target_character', aggregation_target, 'result']], tiled_answerability_list_qas[['target_character', aggregation_target, 'result']], tiled_accessibility_list_qas[['target_character', aggregation_target, 'result']]])
        report[target_scenario+':set:ALL_character'] = df_for_all_character_metric.groupby([aggregation_target, 'target_character'])['result'].all().mean()
        df_for_character_consistency = pd.concat([binary_qas[['target_character', aggregation_target, 'binarized_model_answer']], tiled_answerability_list_qas[['target_character', aggregation_target, 'binarized_model_answer']], tiled_accessibility_list_qas[['target_character', aggregation_target, 'binarized_model_answer']]])
        report[target_scenario+':set:character_answer_consistency'] = df_for_character_consistency.groupby([aggregation_target, 'target_character'])['binarized_model_answer'].nunique().eq(1).mean() # how often the model gives the "same answer" for the binary and list questions for the same character

        for k, v in report.items():
            if isinstance(v, float):
                report[k] = round(v, 3) * 100

        return report

    def run_reports(self, qa_results):
        """
        Create report after scoring and analyzing the results

        Input:
        - qa_results: a list of qa results

        Output:
        - report: a dictionary of scores and analysis

        Note:
        We can further increase the difficulty of the task by changing the aggregation target from 'set_id' to 'part_id' or 'conversation_id'.
        A conversation part refers to the brief section of the conversation that is the relevant part to the question.
        Each conversation part comprises multiple sets of questions, and every conversation consists of multiple conversation parts.
        For instance, if you designate 'part_id' as the aggregation target, the ALL scores will be aggregated for each individual part of the conversation.
        This adjustment will result in the ALL score being aggregated across multiple sets.

        Currently, the default conversation-input-type is 'short' and the ALL scores are aggregated for each set of questions (i.e., aggregation-target to 'set'), which will be the easiest setup for the models.
        The most difficult setup will be to give the full conversation input to the model (i.e., conversation-input-type to 'full') and aggregate the ALL scores for each conversation (i.e., aggregation-target to 'conversation_id')
        """
        df = pd.DataFrame(qa_results)

        # Drop binary questions with no:long answer when input type is short
        if self.args.conversation_input_type == "short":
            df.drop(df[(df['question_type'].str.endswith(":binary")) & (df['correct_answer'] == 'no:long')].index, inplace=True)

        df['conversation_id'] = df['set_id'].map(lambda x: x.split("-")[0])
        df['part_id'] = df['set_id'].map(lambda x: "-".join(x.split("-")[:2]))

        report = self.score_and_analyze(df, target_scenario='inaccessible')
        control_question_report = self.score_and_analyze(df, target_scenario='accessible')
        reports = {'fantom': report, 'control_task': control_question_report}

        print("\n[[ FANToM input type: {} ]]".format(self.args.conversation_input_type))
        print("[[ Model: {} ]]\n".format(self.args.model))
        for k, v in reports['fantom'].items():
            print(k, ":", v)
            print()

        return reports

    def dump_report_outputs(self, reports, evaluation_outputs):
        """
        Dump the reports and the evaluation outputs
        """

        evaluated_responses_filename = "evaluated_responses" + self.output_filename_suffix
        output_dict = {'model': self.args.model, 'results': evaluation_outputs}
        os.makedirs(EVAL_DIR_PATH, exist_ok=True)
        with open(os.path.join(EVAL_DIR_PATH, evaluated_responses_filename), 'w') as f:
            json.dump(output_dict, f, indent=4)

        controlq_report_filename = "control_task_report" + self.output_filename_suffix
        with open(os.path.join(EVAL_DIR_PATH, controlq_report_filename), 'w') as f:
            json.dump(reports['control_task'], f, indent=4)

        report_filename = "REPORT" + self.output_filename_suffix
        with open(os.path.join(EVAL_DIR_PATH, report_filename), 'w') as f:
            json.dump(reports['fantom'], f, indent=4)

        print(">>>>> Dumped evaluation outputs and the report at {}!".format(EVAL_DIR_PATH))
        print(">>>>> Evaluated model responses filename: {}".format(evaluated_responses_filename))
        print(">>>>> REPORT filename: {}".format(report_filename))

    def set_beliefQA_multiple_choices(self, qa):
        if qa['question_type'].endswith(":inaccessible"):
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']
        else:
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']

        answer_goes_last = random.choice([True, False])
        if answer_goes_last:
            choices = [option_a, option_b]
            answer = 1
        else:
            choices = [option_b, option_a]
            answer = 0

        # option letters iterate over the alphabet
        option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
        choices_text = ""
        for letter, option in zip(option_letters, choices):
            choices_text += "{} {}\n".format(letter, option)

        return choices_text, answer

    def setup_fantom(self):
        """
        Flatten the dictionary and add short and full conversation context to each question.
        The result will be a list of questions and list of short or full inputs to be used as input for the models.
        """
        if self.args.aggregation_target == "conversation":
            assert self.args.conversation_input_type == "full", "The input type should have been the full conversation. It doesn't make sense to aggregate the scores over the full conversation when the input is not the full conversation"

        self.fantom_df_to_run = self.fantom_df

        total_num_q = 0
        for idx, _set in self.fantom_df_to_run.iterrows():
            total_num_q += len(_set['beliefQAs'])
            total_num_q += len(_set['answerabilityQAs_binary'])
            total_num_q += len(_set['infoAccessibilityQAs_binary'])
            if _set['factQA'] is not None:
                total_num_q += 1
            if _set['answerabilityQA_list'] is not None:
                total_num_q += 1
            if _set['infoAccessibilityQA_list'] is not None:
                total_num_q += 1

        inputs = []
        qas = []
        for idx, _set in self.fantom_df_to_run.iterrows():
            if self.args.conversation_input_type == "short":
                context = _set['short_context'].strip()
            elif self.args.conversation_input_type == "full":
                context = _set['full_context'].strip()
            
            set_id = _set['set_id']
            fact_q = _set['factQA']['question']
            fact_a = _set['factQA']['correct_answer']

            # Fact Question
            _set['factQA']['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
            _set['factQA']['input_text'] = input_text
            _set['factQA']['set_id'] = set_id
            qas.append(_set['factQA'])
            inputs.append(input_text)

            for _belief_qa in _set['beliefQAs']:
                # Belief Questions
                _belief_qa['context'] = context
                input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, _belief_qa['question'])
                _belief_qa['input_text'] = input_text
                _belief_qa['set_id'] = set_id
                qas.append(_belief_qa)
                inputs.append(input_text)

                # Multiple Choice Belief Questions
                _mc_belief_qa = {**_belief_qa}
                choices_text, answer = self.set_beliefQA_multiple_choices(_mc_belief_qa)
                mc_question = "{}\n{}\n\nChoose an answer from above:".format(_belief_qa['question'], choices_text.strip())
                _mc_belief_qa['question'] = mc_question
                _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
                _mc_belief_qa['choices_text'] = choices_text
                _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
                _mc_belief_qa['correct_answer'] = answer
                input_text = "{}\n\nQuestion: {}".format(context, mc_question)
                _mc_belief_qa['input_text'] = input_text
                qas.append(_mc_belief_qa)
                inputs.append(input_text)

            # Answerability List Questions
            _set['answerabilityQA_list']['fact_question'] = fact_q
            _set['answerabilityQA_list']['context'] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(context, fact_q, _set['answerabilityQA_list']['question'])
            _set['answerabilityQA_list']['input_text'] = input_text
            _set['answerabilityQA_list']['set_id'] = set_id
            if self.args.conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
                _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['answerabilityQA_list'])
            inputs.append(input_text)

            # Answerability Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _answerability_qa in _set['answerabilityQAs_binary']:
                _answerability_qa['fact_question'] = fact_q
                _answerability_qa['context'] = context
                input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, _answerability_qa['question'])
                _answerability_qa['input_text'] = input_text
                _answerability_qa['set_id'] = set_id
                if self.args.conversation_input_type == "full":
                    _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_answerability_qa)
                inputs.append(input_text)

            # Info Accessibility List Questions
            _set['infoAccessibilityQA_list']['fact_question'] = fact_q
            _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
            _set['infoAccessibilityQA_list']['context'] = context
            input_text = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(context, fact_q, fact_a, _set['infoAccessibilityQA_list']['question'])
            _set['infoAccessibilityQA_list']['input_text'] = input_text
            _set['infoAccessibilityQA_list']['set_id'] = set_id
            if self.args.conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
                _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['infoAccessibilityQA_list'])
            inputs.append(input_text)

            # Info Accessibility Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                _info_accessibility_qa['fact_question'] = fact_q
                _info_accessibility_qa['fact_answer'] = fact_a
                _info_accessibility_qa['context'] = context
                input_text = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, fact_a, _info_accessibility_qa['question'])
                _info_accessibility_qa['input_text'] = input_text
                _info_accessibility_qa['set_id'] = set_id
                if self.args.conversation_input_type == "full":
                    _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_info_accessibility_qa)
                inputs.append(input_text)

        self.inputs = inputs
        self.flattened_fantom = qas

    def parse_response(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip()

        return response

    def get_last_savepoint(self):
        responses_filename = "model_responses" + self.output_filename_suffix + "l" # jsonl
        model_responses_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)

        # check if model outputs file exists
        if os.path.exists(model_responses_filename_path):
            print("File {} exists. Reading responses from file...".format(model_responses_filename_path))
            df = pd.read_json(model_responses_filename_path, lines=True)
            if len(df) > 0:
                last_idx = df.iloc[-1]['index']
                model_responses = df['response'].tolist()
            else:
                last_idx = -1
                model_responses = []
        else:
            last_idx = -1
            model_responses = []
        
        return last_idx, model_responses, model_responses_filename_path

    def run_batch_inference(self):
        fantom_dataset = FantomDataset(self.inputs, self.args)
        loader = DataLoader(fantom_dataset, batch_size=self.args.batch_size)

        model_responses = []
        print("Generating responses...")
        last_idx, model_responses, response_filename_path = self.get_last_savepoint()
        if last_idx > 0:
            last_idx = last_idx // self.args.batch_size
        for batch_idx, batch in enumerate(tqdm(loader)):
            if batch_idx <= last_idx:
                continue

            if self.args.use_cot:
                batch = [b.removesuffix("Answer:") + " Let's think step by step." for b in batch]
                _cot_response = self.model.batch_interact(batch)
                cot_response = self.parse_response(_cot_response)
                for bidx, b in enumerate(batch):
                    batch[bidx] = b + " " + cot_response[bidx] + "\n\nTherefore, the answer is:"

            responses = self.model.batch_interact(batch)

            for idx, response in enumerate(responses):
                response = self.parse_response(response)
                model_responses.append(response)

                # save the model responses in a file on the fly
                with open(response_filename_path, 'a') as f:
                    instance_for_dump = {'index': batch_idx * self.args.batch_size + idx, 'response': response, 'input_prompt': batch[idx]}
                    json.dump(instance_for_dump, f)
                    f.write("\n")

        return model_responses

    def run_inference(self):
        target_data = self.inputs
        model_responses = []

        # check if the file exists
        last_idx, model_responses, response_filename_path = self.get_last_savepoint()

        print("Generating responses...")
        for idx, input_prompt in enumerate(tqdm(target_data)):
            if idx <= last_idx:
                continue

            if self.args.use_cot:
                cot_input_prompt = input_prompt + " Let's think step by step."
                cot_response = self.model.interact(cot_input_prompt)
                cot_response = self.parse_response(cot_response)
                input_prompt = cot_input_prompt + " " + cot_response + "\n\nTherefore, the answer is:"
            response = self.model.interact(input_prompt)
            response = self.parse_response(response)
            model_responses.append(response)

            # save the model responses in a file on the fly
            with open(response_filename_path, 'a') as f:
                json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
                f.write("\n")

        return model_responses

    def run(self):
        os.makedirs(EVAL_DIR_PATH, exist_ok=True)
        if args.existing_response_file_name is None:
            if self.args.model.startswith("gpt-") or self.args.model.startswith("text-") or self.args.model.endswith("-tg"):
                model_responses = self.run_inference()
            else:
                model_responses = self.run_batch_inference()
        else:
            print(">>> Reading responses from file...")
            model_responses = self.get_responses_from_file(self.args.existing_response_file_name)

        evaluated_outputs = self.evaluate_response(self.flattened_fantom, model_responses)
        reports = self.run_reports(evaluated_outputs)
        self.dump_report_outputs(reports, evaluated_outputs)

    def get_responses_from_file(self, response_filename):
        setup = response_filename.removeprefix("model_responses").removesuffix(".jsonl")
        assert setup == self.output_filename_suffix.removesuffix(".json"), "The response file name does not match the output file name"

        response_file = os.path.join(EVAL_DIR_PATH, response_filename)
        df = pd.read_json(response_file, lines=True)
        model_responses = df['response'].to_list()
        return model_responses

def main(args):
    evaluator = FantomEvalAgent(args)
    evaluator.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4-0314',
                        choices=['gpt-4-1106-preview', 'gpt-4-0613', 'gpt-4-0314', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301', 'text-davinci-003', 'text-davinci-002', 'text-curie-001', 'flan-ul2', 'flan-t5-xxl', 'flan-t5-xl', 'Llama-2-13b-hf', 'Llama-2-13b-chat-hf', 'llama-2-70b-tg', 'llama-2-70b-chat-tg', 'zephyr-7b-alpha', 'zephyr-7b-beta', 'mistral', 'mistral-instruct', 'mpt-30b-instruct-tg', 'guanaco-33b-tg'],
                        help='name of the model to run evaluation',
    )
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='batch size for evaluation',
    )
    parser.add_argument('--conversation-input-type',
                        type=str,
                        default='short',
                        choices=['short', 'full'],
                        help='whether to use short or full conversation input',
    )
    parser.add_argument('--aggregation-target',
                        type=str,
                        default='set',
                        choices=['set', 'part', 'conversation'],
                        help='whether to aggregate the ALL scores at the set, part, or conversation level. As the level increases, the task will be more difficult',
    )
    parser.add_argument('--existing-response-file-name',
                        type=str,
                        help='name of the response file that you want to recompute the report for',
    )
    parser.add_argument('--use-cot',
                        type=bool,
                        default=False,
                        help='whether to use cot or not',
    )
    args = parser.parse_args()
    main(args)
