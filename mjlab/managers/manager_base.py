import abc


class ManagerTermBase(abc.ABC):
  def __call__(self):
    pass


class ManagerBase(abc.ABC):
  """Base class for all managers."""

  def __init__(self):
    pass

  def reset(self):
    pass
