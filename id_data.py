from typing import Generic, NamedTuple, TypeVar

K = TypeVar("K")
class _IdData(NamedTuple):
    id:int
    data:K
class IdData(_IdData,Generic[K]):
    pass;