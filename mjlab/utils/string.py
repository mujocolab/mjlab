import re
from typing import Dict, List, Pattern, Any, Tuple, Sequence


def resolve_expr(
  pattern_map: Dict[str, Any],
  names: List[str],
  default_val: Any = 0.0,
) -> List[Any]:
  # Pre-compile patterns in insertion order.
  compiled: List[Tuple[Pattern[str], Any]] = [
    (re.compile(pat), val) for pat, val in pattern_map.items()
  ]

  result: List[Any] = []
  for name in names:
    for pat, val in compiled:
      if pat.match(name):
        result.append(val)
        break
    else:
      result.append(default_val)
  return result


def filter_exp(exprs: List[str], names: List[str]) -> List[str]:
  patterns: List[Pattern] = [re.compile(expr) for expr in exprs]
  return [name for name in names if any(pat.match(name) for pat in patterns)]


def resolve_field(field: int | dict[str, int], names: list[str], default_val: Any = 0):
  return (
    resolve_expr(field, names, default_val)
    if isinstance(field, dict)
    else [field] * len(names)
  )


def resolve_matching_names(
  keys: str | Sequence[str],
  list_of_strings: Sequence[str],
  preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
  # resolve name keys
  if isinstance(keys, str):
    keys = [keys]
  # find matching patterns
  index_list = []
  names_list = []
  key_idx_list = []
  # book-keeping to check that we always have a one-to-one mapping
  # i.e. each target string should match only one regular expression
  target_strings_match_found = [None for _ in range(len(list_of_strings))]
  keys_match_found = [[] for _ in range(len(keys))]
  # loop over all target strings
  for target_index, potential_match_string in enumerate(list_of_strings):
    for key_index, re_key in enumerate(keys):
      if re.fullmatch(re_key, potential_match_string):
        # check if match already found
        if target_strings_match_found[target_index]:
          raise ValueError(
            f"Multiple matches for '{potential_match_string}':"
            f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
          )
        # add to list
        target_strings_match_found[target_index] = re_key
        index_list.append(target_index)
        names_list.append(potential_match_string)
        key_idx_list.append(key_index)
        # add for regex key
        keys_match_found[key_index].append(potential_match_string)
  # reorder keys if they should be returned in order of the query keys
  if preserve_order:
    reordered_index_list = [None] * len(index_list)
    global_index = 0
    for key_index in range(len(keys)):
      for key_idx_position, key_idx_entry in enumerate(key_idx_list):
        if key_idx_entry == key_index:
          reordered_index_list[key_idx_position] = global_index
          global_index += 1
    # reorder index and names list
    index_list_reorder = [None] * len(index_list)
    names_list_reorder = [None] * len(index_list)
    for idx, reorder_idx in enumerate(reordered_index_list):
      index_list_reorder[reorder_idx] = index_list[idx]
      names_list_reorder[reorder_idx] = names_list[idx]
    # update
    index_list = index_list_reorder
    names_list = names_list_reorder
  # check that all regular expressions are matched
  if not all(keys_match_found):
    # make this print nicely aligned for debugging
    msg = "\n"
    for key, value in zip(keys, keys_match_found):
      msg += f"\t{key}: {value}\n"
    msg += f"Available strings: {list_of_strings}\n"
    # raise error
    raise ValueError(
      f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
    )
  # return
  return index_list, names_list


def resolve_matching_names_values(
  data: dict[str, Any], list_of_strings: Sequence[str], preserve_order: bool = False
) -> tuple[list[int], list[str], list[Any]]:
  """Match a list of regular expressions in a dictionary against a list of strings and return
  the matched indices, names, and values.

  If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
  of the provided list of strings. This means that the ordering is dictated by the order of the target strings
  and not the order of the query regular expressions.

  If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
  of the provided list of query regular expressions.

  For example, consider the dictionary is {"a|d|e": 1, "b|c": 2}, the list of strings is ['a', 'b', 'c', 'd', 'e'].
  If :attr:`preserve_order` is False, then the function will return the indices of the matched strings, the
  matched strings, and the values as: ([0, 1, 2, 3, 4], ['a', 'b', 'c', 'd', 'e'], [1, 2, 2, 1, 1]). When
  :attr:`preserve_order` is True, it will return them as: ([0, 3, 4, 1, 2], ['a', 'd', 'e', 'b', 'c'], [1, 1, 1, 2, 2]).

  Args:
      data: A dictionary of regular expressions and values to match the strings in the list.
      list_of_strings: A list of strings to match.
      preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.

  Returns:
      A tuple of lists containing the matched indices, names, and values.

  Raises:
      TypeError: When the input argument :attr:`data` is not a dictionary.
      ValueError: When multiple matches are found for a string in the dictionary.
      ValueError: When not all regular expressions in the data keys are matched.
  """
  # check valid input
  if not isinstance(data, dict):
    raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")
  # find matching patterns
  index_list = []
  names_list = []
  values_list = []
  key_idx_list = []
  # book-keeping to check that we always have a one-to-one mapping
  # i.e. each target string should match only one regular expression
  target_strings_match_found = [None for _ in range(len(list_of_strings))]
  keys_match_found = [[] for _ in range(len(data))]
  # loop over all target strings
  for target_index, potential_match_string in enumerate(list_of_strings):
    for key_index, (re_key, value) in enumerate(data.items()):
      if re.fullmatch(re_key, potential_match_string):
        # check if match already found
        if target_strings_match_found[target_index]:
          raise ValueError(
            f"Multiple matches for '{potential_match_string}':"
            f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
          )
        # add to list
        target_strings_match_found[target_index] = re_key
        index_list.append(target_index)
        names_list.append(potential_match_string)
        values_list.append(value)
        key_idx_list.append(key_index)
        # add for regex key
        keys_match_found[key_index].append(potential_match_string)
  # reorder keys if they should be returned in order of the query keys
  if preserve_order:
    reordered_index_list = [None] * len(index_list)
    global_index = 0
    for key_index in range(len(data)):
      for key_idx_position, key_idx_entry in enumerate(key_idx_list):
        if key_idx_entry == key_index:
          reordered_index_list[key_idx_position] = global_index
          global_index += 1
    # reorder index and names list
    index_list_reorder = [None] * len(index_list)
    names_list_reorder = [None] * len(index_list)
    values_list_reorder = [None] * len(index_list)
    for idx, reorder_idx in enumerate(reordered_index_list):
      index_list_reorder[reorder_idx] = index_list[idx]
      names_list_reorder[reorder_idx] = names_list[idx]
      values_list_reorder[reorder_idx] = values_list[idx]
    # update
    index_list = index_list_reorder
    names_list = names_list_reorder
    values_list = values_list_reorder
  # check that all regular expressions are matched
  if not all(keys_match_found):
    # make this print nicely aligned for debugging
    msg = "\n"
    for key, value in zip(data.keys(), keys_match_found):
      msg += f"\t{key}: {value}\n"
    msg += f"Available strings: {list_of_strings}\n"
    # raise error
    raise ValueError(
      f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
    )
  # return
  return index_list, names_list, values_list
