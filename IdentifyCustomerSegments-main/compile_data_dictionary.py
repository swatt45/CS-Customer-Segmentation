#!/usr/bin/env python3.10.4

"""uses Data_Dictionary.md and AZDIAS_Feature_Summary.csv to create a comprehensive
view of individual features"""

import re
from typing import Iterator, List, Match, Tuple, Dict
import pandas as pd
import IPython  # needed for display to work

__author__ = "Jason Secrest"

# %%

DataFrame = type(pd.DataFrame())
Series = type(pd.Series(dtype="object"))

FEATURE_NAMES_GROUP_INDEX = 1
DEFINITION1_GROUP_INDEX = 2
VALUES_GROUP_INDEX = 3
DESCRIPTION2_GROUP_INDEX = 4


def _get_desc_iter(filepath: str) -> Iterator[Match[str]]:
    """return a 'description iterator' that returns descriptive sections of the data,
    each section representing one feature.

    Parameters
    ----------
        filepath : path to Data_Dictionary.md
    """

    with open(filepath, "r", encoding="UTF-8") as dd_file:
        doc = dd_file.read()

    desc_str = (
        r"((?:\#{3}[^\n]*\n)+)((?:(?:.*)\n)(?:(?:[^-\n]*)*\n)?)"
        + r"((?:-[ -]+(?:[^\n]*)\n(?: {5}.*\n)?)*)?"
        + r"(?:\n(Dimension translations:\n(?:- .*\n)*))?"
    )
    desc_pat = re.compile(desc_str)
    desc_iter = desc_pat.finditer(doc)
    return desc_iter


# %%
def _is_empty_group(group) -> bool:
    """determines if the result a match is empty

    Parameters
    ----------
        group : a group from a regex match

    Returns
    -------
        bool : True if the match is empty or None
    """
    empty_string = r"^\s*$"
    if (group is None) or re.match(pattern=empty_string, string=group) is not None:
        return True


def _get_feature_names(desc_match: Match) -> List[str] | None:
    """return a list of feature names from the data dictionary

    Parameters
    ----------
        desc_match : a match object representing the features included in this feature section

    Returns
    -------
        List[str] : a list of feature names
    """
    feature_group = desc_match.group(FEATURE_NAMES_GROUP_INDEX)
    if _is_empty_group(feature_group):
        return None
    feature_names_str = r"(?:\d+\.\d+\. )?(\w+)"
    section_no_str = r"(\d+\.\d+\.)"
    feature_names = re.findall(pattern=feature_names_str, string=feature_group)
    section_no = [re.search(pattern=section_no_str, string=feature_group).group(1)]
    for _ in range(1, len(feature_names)):
        section_no.append(section_no[0])
    if len(feature_names) != len(section_no):
        raise ValueError("section_no is not the same length as feature_names")
    return feature_names, section_no


def _get_definition(desc_match: Match) -> List[str] | None:
    """extract the definition of this section of the data dictionary

    Parameters
    ----------
        desc_match : a match object representing the features included in this feature section

    Returns
    -------
        List[str] : a list of len 1 containing a string with the dictionary entry
    """
    def_group = desc_match.group(DEFINITION1_GROUP_INDEX)
    if _is_empty_group(def_group):
        return None
    def_lines_str = r"([^\n]*)\n"
    def_lines = re.findall(pattern=def_lines_str, string=def_group)
    definition = " ".join(def_lines)
    return [definition]


def get_codes(desc_match: Match) -> [List[DataFrame] | None, List[str] | None]:
    """returns a pandas DataFrame containing codes strings and code definition strings.
    - Codes are the feature values from the data. A number or letter like "-1" or "W"
    - Code definitions are the meanings of the values.

    Parameters
    ----------
        desc_match : a match object representing the features included in this feature section
    """
    match_group: str = desc_match.group(VALUES_GROUP_INDEX)
    if _is_empty_group(match_group):
        return [None, None]
    # get the VALUE and the VALUE_DESCRIPTION
    # from example line "-  2: very likely"
    # value is 2, value_description is "very likely"
    symbols_and_defs_str = r"- {1,2}((?:-?\d+)?(?:[a-zA-Z])?): (.*)\n((?: {5}.*\n)*)"
    symbols_and_defs: List[Tuple(str, str)] | Tuple(str, str, str) = re.findall(
        symbols_and_defs_str, match_group
    )
    if symbols_and_defs is None:
        # deal with "- missing data encoded as 0"
        symbols_and_defs_str = "- (.*)\n"
        symbols_and_defs = re.findall(symbols_and_defs_str, match_group)
        if symbols_and_defs:
            symbols_and_defs = [("0", "missing")]
    if symbols_and_defs is None:
        raise ValueError(
            "No definition found in match.group({VALUES_GROUP_INDEX}): {g}"
        )

    # In a Tuple T in the list,
    #   T[0] is a symbol like "0" or "W" or "-1"
    #   T[1] is the symbol_definition associated with the symbol
    #   T[2] (if it exists) is any number overflow lines from the definition
    #       - Each overflow line starts with 5 spaces and ends with \n

    # process each tuple
    # separate
    symbols = []
    definitions = []
    for feature_tuple in symbols_and_defs:
        #   When T[2] exists, get its definition values and merge them with T[1]
        definition: str = None
        if len(feature_tuple) == 3:
            extra_lines: List[str] = re.findall(
                pattern=" {5}([^\n]+)", string=feature_tuple[2]
            )
            extra_lines.insert(0, feature_tuple[1])
            definition = " ".join(extra_lines)
        else:
            if len(feature_tuple) != 2:
                raise ValueError(
                    f"Expected tuple of len 2 or 3. Got len: {len(feature_tuple)}"
                )
            definition.append(feature_tuple[1])
        if definition is None:
            raise ValueError("var 'definition' not set")
        symbols.append(feature_tuple[0])
        definitions.append(definition)
        # print(symbols, definitions)

    codes_s = pd.Series(definitions, index=symbols, dtype="object")
    allowed_values = codes_s.index.tolist()
    return [[codes_s], [allowed_values]]


def _get_dim_translate(desc_match: Match) -> List[str] | None:
    """return a list of translation relevant to the dimensions of a feature section
    (Most features do not have a dimensional translation)

    Parameters
    ----------
        desc_match : a match object representing the features included in this feature section

    Returns
    -------
        List[str] : a list of dimensional translations
    """
    desc_group = desc_match.group(DESCRIPTION2_GROUP_INDEX)
    if _is_empty_group(desc_group):
        return None
    def_lines_str = r"- (\w*.+)\n?"
    dimensional_translations = re.findall(pattern=def_lines_str, string=desc_group)
    return dimensional_translations


def _get_section_df(desc_match: Match[str]) -> DataFrame | None:
    """return a pandas DataFrame composed of the elements of a feature section in the
    data dictionary

    Parameters
    ----------
        desc_match : a match object representing the features included in this feature
            section
    """
    # only try to process this match if it isn't the table of contents
    if re.match("### Table of Contents", desc_match.group(1)) is None:
        info_dict = {
            "feature_name": _get_feature_names(desc_match)[0],
            "section_no": _get_feature_names(desc_match)[1],
            "definition": _get_definition(desc_match),
            "codes": get_codes(desc_match)[0],
            "allowed_values": get_codes(desc_match)[1],
            "dim_translation": _get_dim_translate(desc_match),
        }
        # populate per feature lists, so lists are same length
        col_count = len(info_dict["feature_name"])
        if col_count > 1:
            # index 0 is already set, just need to fill remaining
            for i in range(1, col_count):
                if i >= col_count:
                    raise ValueError("i is too large. i : {i}, col_count : {col_count}")
                info_dict["definition"].append(info_dict["definition"][0])
                info_dict["codes"].append(info_dict["codes"][0])
                info_dict["allowed_values"].append(info_dict["allowed_values"][0])
        # if the dict didn't contain any columns return none
        out_df: pd.DataFrame() = None
        if col_count > 0:
            out_df = pd.DataFrame(data=info_dict)
            # out_df = out_df.set_index(["feature_name"])
        return out_df


def _next_match(desc_iter, verbose=False):
    """returns a DataFrame from the next match in desc_iter"""
    section_df = None
    while section_df is None:
        match = next(desc_iter)
        section_df = _get_section_df(match)
        if verbose:
            if section_df is None:
                print("Match does not contain feature data.")
    if verbose:
        # display(section_df)
        # display(section_df.info())
        # display(section_df.definition)
        print(section_df.head())
        # print(section_df.loc[:, "codes"])
    return section_df


def _get_data_dict_as_df(filename) -> DataFrame:
    """converts the Data_Dictionary.md file into a dataframe"""
    data_df: DataFrame = None
    desc_iter = _get_desc_iter(filename)
    for match in desc_iter:
        section_df = _get_section_df(match)
        if section_df is not None:
            if data_df is None:
                data_df = section_df
            else:
                data_df = pd.concat([data_df, section_df])
    if data_df is None:
        print(
            "Not able to create DataFrame from data dictionary file (No appropriate regex matches)"
        )
    return data_df


def _get_feature_summary_as_df(filepath) -> DataFrame:
    """converts the AZDIAS_Feature_Summary.csv file into a dataframe and fixes values"""
    fsum_df: DataFrame = pd.read_csv(filepath, sep=";")
    fsum_df = fsum_df.rename(columns={"attribute": "feature_name"})

    # convert list strings  (e.g. list_string : str = "[1,3,5]")
    # into lists of strings (e.g  string_list : List[str] = ["1","3","5"])
    def fix_missing_or_unknown(list_as_str: str):
        list_of_str = list_as_str.strip("[]").split(",")
        if list_of_str == [""]:
            list_of_str = None
        return list_of_str

    fsum_df.loc[:, "missing_or_unknown"] = fsum_df.loc[:, "missing_or_unknown"].apply(
        fix_missing_or_unknown
    )

    return fsum_df


# %%
class DataCodex:
    """Array with associated photographic information.

    Parameters
    ----------
        data_dict_file : relative file path to Data_Dictionary.md
        feat_summary_file : relative file path to AZDIAS_Feature_Summary.csv

    Attributes
    ----------
        all_df {DataFrame} : a dataframe representing both the data
            dictionary and feature summary information for each feature
        feature_names {List[str]} : a list of feature names from `all_df`

    Methods
    -------
        #### get information about a feature as a data structure
        - get_feature_as_s(self, feature_name) -> Series | None
        - get_feature_as_df(self, feature_name) -> DataFrame | None
        - get_feature_as_dict(self, feature_name) -> Dict | None

        ### print or display (in a notebook) information about a feature
        - nice_print_feature(self, feature_name) -> None
        - nice_display_feature(self, feature_name) -> None


    """

    def __init__(self, data_dict_file: str, feat_summary_file: str):
        data_df: DataFrame = _get_data_dict_as_df(data_dict_file)
        feat_sum_df: DataFrame = _get_feature_summary_as_df(feat_summary_file)
        self.all_df: DataFrame = data_df.merge(
            feat_sum_df, how="left", on="feature_name"
        )
        self.all_df.set_index(["feature_name"], inplace=True, drop=False)
        self.all_df = self.all_df.apply(self._trim_allowed_values, axis=1)

    @property
    def feature_names(self) -> List[str]:
        """return a list of features from all_df"""
        return self.all_df.index.to_list()

    def _trim_allowed_values(self, feature_s: Series):
        """intended to be applied to each feature to convert allowed_values
        to a set that excludes missing values"""
        allowed_values = feature_s.loc["allowed_values"]
        missing_values = feature_s.loc["missing_or_unknown"]
        if allowed_values and missing_values:
            allowed_values_set = set(allowed_values)
            missing_values_set = set(missing_values)
            true_allowed_set = allowed_values_set.difference(missing_values_set)
            feature_s.loc["allowed_values"] = true_allowed_set
        return feature_s

    def is_feature_in_data(self, feature_name) -> bool:
        """returns a bool: True if the feature_name is in the features list"""
        return feature_name in self.feature_names

    # METHODS FOR GETTING FEATURES

    def get_feature_as_s(self, feature_name) -> Series | None:
        """returns a pandas Series representation of all the attributes of this feature
        (or None if it doesn't exist)"""
        if self.is_feature_in_data(feature_name) is False:
            raise ValueError(f"'{feature_name}' is not in DataCodex.all_df")
        raw_feature_s = self.all_df.xs(feature_name)
        return raw_feature_s

    def get_feature_as_df(self, feature_name) -> DataFrame | None:
        """returns a pandas DataFrame representation of all the attributes of this feature
        (or None if it doesn't exist)"""
        return self.get_feature_as_s(feature_name).to_frame()

    def get_feature_as_dict(self, feature_name) -> Dict | None:
        """returns a dictionary representation of all the attributes of this feature
        (or None if it doesn't exist)"""
        return self.get_feature_as_s(feature_name).to_dict()

    # METHODS FOR DISPLAYING FEATURES

    def print_feature(self, feature_name) -> None:
        """prints a text-console friendly representation of this feature"""
        if self.is_feature_in_data(feature_name) is False:
            print(
                f"The feature named '{feature_name}' does not appear in the data documentation"
            )
        feature_dict = self.get_feature_as_dict(feature_name)
        for key, value in feature_dict.items():
            if key != "codes":
                print(f"{key:>20}: {value}")
        print("CODES:")
        print(feature_dict["codes"])

    def display_feature(self, feature_name) -> None:
        """displays a jupyter notebook friendly representation of this feature"""

        if self.is_feature_in_data(feature_name) is False:
            if self.is_feature_in_data(feature_name) is False:
                display(
                    f"The feature named '{feature_name}' does not appear in the data documentation"
                )
        feature_s = self.get_feature_as_s(feature_name)
        partial_df = feature_s.drop(index=["codes"]).to_frame()
        partial_df.index.name = "attribute"
        partial_df.columns = ["value"]
        display(partial_df)
        code_df = feature_s["codes"]
        if code_df is not None:
            code_df = feature_s["codes"].to_frame()
            code_df.index.name = "code"
            code_df.columns = ["description"]
            display(code_df)
        else:
            display("CODES: None")


# %%
if __name__ == "__main__":

    # testing

    codex = DataCodex(
        data_dict_file="data/Data_Dictionary.md",
        feat_summary_file="data/AZDIAS_Feature_Summary.csv",
    )

    # print(data_codex.feature_names)
    # %%
    FEATURE_TESTS = [
        "FINANZ_MINIMALIST",
        "ANZ_PERSONEN",
        "CAMEO_INTL_2015",
        "OST_WEST_KZ",
        "ANZ_HAUSHALTE_AKTIV",
        "HH_EINKOMMEN_SCORE",
    ]
    for test in FEATURE_TESTS:
        codex.print_feature(test)
        ## for use in jupyter notebook or interactive python ##
        codex.display_feature(test)
    # %%
