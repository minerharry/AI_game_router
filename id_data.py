from typing import Generic, NamedTuple, TypeVar, TYPE_CHECKING

K = TypeVar("K")
if TYPE_CHECKING:
    class IdData(NamedTuple,Generic[K]):
        id:int
        data:K
else:
    class _IdData(NamedTuple):
        id:int
        data:K
    class IdData(_IdData,Generic[K]):
        pass;