"""
This code is taken directly from the following repository:
https://github.com/gucci-j/negotiation-breakdown-detection/tree/main

@inproceedings{yamaguchi-etal-2021-dialogue,
    title = "Dialogue Act-based Breakdown Detection in Negotiation Dialogues",
    author = "Yamaguchi, Atsuki  and
      Iwasa, Kosui  and
      Fujita, Katsuhide",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = https://aclanthology.org/2021.eacl-main.63,
    pages = "745--757",
    abstract = "Thanks to the success of goal-oriented negotiation dialogue systems, studies of negotiation dialogue have gained momentum in terms of both human-human negotiation support and dialogue systems. However, the field suffers from a paucity of available negotiation corpora, which hinders further development and makes it difficult to test new methodologies in novel negotiation settings. Here, we share a human-human negotiation dialogue dataset in a job interview scenario that features increased complexities in terms of the number of possible solutions and a utility function. We test the proposed corpus using a breakdown detection task for human-human negotiation support. We also introduce a dialogue act-based breakdown detection method, focusing on dialogue flow that is applicable to various corpora. Our results show that our proposed method features comparable detection performance to text-based approaches in existing corpora and better results in the proposed dataset.",
}
"""


import json
import itertools
from pathlib import Path
from typing import List, TypeVar, Dict
from datetime import datetime as dt

Option = TypeVar('Option', str, int)


class Bid():
    def __init__(self, hash_id: str, user, options: Dict[str, Option], accepted: bool, created_at: float):
        self.hash_id = hash_id
        self.user = user
        self.options = options
        self.accepted = accepted
        self.created_at = created_at


class DiscreteIssue():
    """
    The instance of issue which takes options that is `str`

    For example: workplace, company, and so on.
    """

    def __init__(self, name: str, options: List[Option], related_issue=None):
        self.name = name
        self.options = options
        self.related_issue = related_issue


class IntegerIssue():
    """
    The instance of issue which takes options that is `int`

    For example: salary, weely holiday, and so on.
    """

    def __init__(self, name: str, option_min: int, option_max: int):
        self.name = name
        self.option_min = option_min
        self.option_max = option_max
        self.options = [str(o) for o in range(option_min, option_max+1)]


Issue = TypeVar('Issue', DiscreteIssue, IntegerIssue)


class User():
    """
    The instance of user.

    :params hash_id: an unique id of the user
    :params context: information required for calculating a score for a certain bid
    """

    def __init__(self, hash_id: str, worker_id: str, assignment_id: str, context: dict):
        self.hash_id = hash_id
        self.worker_id = worker_id
        self.assignment_id = assignment_id
        self.context = context

    def calc_score(self, bid: Bid):
        """Calculate a score that the user can earn by the bid."""

        score = 0
        user_role = self.context["role"]
        user_utilities = self.context["utilities"]
        for issue_name, option in bid.options.items():
            issue_utility = None
            for u in user_utilities:
                if u["name"] == issue_name:
                    issue_utility = u
            issue_weight = issue_utility["weight"]

            if issue_utility["type"] == "INTEGER":
                option_max = issue_utility["max"]
                option_min = issue_utility["min"]
                if user_role == "recruiter":
                    score += issue_weight * (option_max - option) / (option_max - option_min)
                elif user_role == "worker":
                    score += issue_weight * (option - option_min) / (option_max - option_min)
                else:
                    raise Exception("No such role: {}".format(user_role))
            elif issue_utility["type"] == "DISCRETE":
                if "relatedTo" in issue_utility:
                    option_weight = None
                    related_issue_name = issue_utility["relatedTo"]
                    for o in issue_utility["options"]:
                        if o["names"][related_issue_name] == bid.options[related_issue_name]\
                                and o["names"][issue_name] == option:
                            option_weight = o["weight"]
                    score += option_weight * issue_weight
                else:
                    option_weight = None
                    for o in issue_utility["options"]:
                        if o["name"] == option:
                            option_weight = o["weight"]
                    score += option_weight * issue_weight
            else:
                raise Exception("No such issue type: {}".format(issue_utility["type"]))
        return score


class Comment():
    def __init__(self, hash_id: str, user: User, body: str, created_at: float):
        self.hash_id = hash_id
        self.user = user
        self.body = body
        self.created_at = created_at


class Negotiation():
    """
    The instance of negotiation.

    :param hash_id: an unique id of the negotiation
    :param status: status of end of the negotiation, which takes 'completed', 'aborted', 'terminated'
    :param users: participants of the negotiation
    :param comments: utterances in the negotiation
    :param bids: bids which is suggested by a user in the negotiation
    """

    def __init__(
            self,
            hash_id: str,
            status: str,
            users: List[User],
            comments: List[Comment],
            issues: List[Issue],
            bids: List[Bid]
    ):
        self.hash_id = hash_id
        self.status = status
        self.users = users
        self.comments = comments
        self.bids = bids
        self.issues = issues

        solution = None
        for bid in self.bids:
            if bid.accepted:
                solution = bid
        self.solution = solution

    def get_all_bids(self):
        """
        Returns all bids which are *possible* to take
        """

        option_lists = []  # List[List[Option]]
        issue_names = []  # List[str]
        for issue in self.issues:
            if isinstance(issue, DiscreteIssue):
                option_lists.append(issue.options)
                issue_names.append(issue.name)
            elif isinstance(issue, IntegerIssue):
                option_lists.append(list(range(issue.option_min, issue.option_max)))
                issue_names.append(issue.name)
        bids = list(itertools.product(*option_lists))
        bids = [Bid(None, None, {name: option for name, option in zip(
            issue_names, options)}, None, None) for options in bids]
        return bids

    def get_pareto_bids(self):
        bids = self.get_all_bids()
        bid_scores_pairs = [(bid, [user.calc_score(bid) for user in self.users]) for bid in bids]

        pareto_bid_scores_pairs = []
        if len(self.users) != 2:
            raise NotImplementedError(
                "We only supports a negotiation with two users. the negotiation has {} users".format(len(self.users)))

        bid_scores_pairs = sorted(bid_scores_pairs, key=lambda s: -s[1][0])

        while True:
            if len(bid_scores_pairs) < 2:
                pareto_bid_scores_pairs += bid_scores_pairs
                break

            top_bid_scores_pair = bid_scores_pairs[0]
            pareto_bid_scores_pairs.append(top_bid_scores_pair)
            _, top_bid_scores = top_bid_scores_pair

            bid_scores_pairs = bid_scores_pairs[1:]
            bid_scores_pairs = [(b, scores)
                                for b, scores in bid_scores_pairs if scores[1] > top_bid_scores[1]]
        bid_scores_pairs = pareto_bid_scores_pairs
        return [b for b, scores in bid_scores_pairs]


def str2timestamp(str_date: str) -> float:
    str_date = str_date[:19]
    date = dt.strptime(str_date, '%Y-%m-%d %H:%M:%S')
    return date.timestamp()


def read_ji_negotiations(file_name: str, breakdown_flag=False) -> List[Negotiation]:
    """
    Read negotiations from file.

    :param file_name: file path which contains negotiations with json format
    """
    negos_raw = json.loads(Path(file_name).read_text())
    negotiations = []

    for nego_raw in negos_raw:
        # Define issues
        salary_issue = IntegerIssue("Salary", 20, 51)
        workplace_issue = DiscreteIssue("Workplace", ["Tokyo", "Seoul", "Beijing", "Sydney"])
        weekly_holiday_issue = IntegerIssue("Weekly holiday", 2, 7)
        company_issue = DiscreteIssue("Company", ["Google", "Facebook", "Apple", "Amazon"])
        position_issue = DiscreteIssue("Position", ["Engineer", "Manager", "Designer", "Sales"], company_issue)
        issues = [
            salary_issue,
            workplace_issue,
            weekly_holiday_issue,
            company_issue,
            position_issue
        ]

        # Create `User` instances
        users_raw = nego_raw["users"]
        users = [User(user_raw["id"], user_raw["worker_id"], user_raw["assignment_id"],
                      {"role": user_raw["role"], "utilities": user_raw["utilities"]}) for user_raw in users_raw]

        # Create `Comment` instances
        id2user = {user.hash_id: user for user in users}
        comments_raw = nego_raw["comments"]
        comments = [Comment(raw["id"], id2user[raw["user_id"]], raw["body"],
                            str2timestamp(raw["created_at"])) for raw in comments_raw]

        # Create `Bid` instances
        bids_raw = nego_raw["solutions"]
        bids = [Bid(bid_raw["id"], id2user[bid_raw["user_id"]], bid_raw["body"],
                    bid_raw["accepted"], str2timestamp(bid_raw["created_at"])) for bid_raw in bids_raw]

        # Create an `Negotiation` instance
        nego_status = nego_raw["status"]
        nego_hash_id = nego_raw["id"]
        negotiation = Negotiation(nego_hash_id, nego_status, users, comments, issues, bids)
        negotiations.append(negotiation)

    if breakdown_flag is True:
        # For Breakdown Detection
        negotiations = [n for n in negotiations if n.status != "aborted"]
    else:
        # For Ordinary Tasks
        negotiations = [n for n in negotiations if n.solution is not None]
        negotiations = [n for n in negotiations if n.status == "completed"]
    
    return negotiations


def main():
    negos = read_ji_negotiations("./data/data.json")
    print("There are {} completed negotiations.".format(len(negos)))


if __name__ == "__main__":
    main()