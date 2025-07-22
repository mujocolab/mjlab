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
