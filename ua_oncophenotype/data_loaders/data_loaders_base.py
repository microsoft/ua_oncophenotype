from typing import Dict, Iterator, List, Optional, Sequence, Union, overload

from ..dtypes import BaseModel, BaseObject, Label


class DatasetBase(BaseModel):
    """Dataset is a simple list of BaseObjects with optional labels.  The basic
    requirement is that each item has an id."""

    items: Sequence[BaseObject]
    labels: Optional[Dict[str, Label]] = None

    def __iter__(self) -> Iterator[BaseObject]:
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    @overload
    def __getitem__(self, key: int) -> BaseObject:
        ...

    @overload
    def __getitem__(self, key: slice) -> "DatasetBase":
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[BaseObject, "DatasetBase"]:
        if isinstance(key, slice):
            items = self.items[key]
            labels = None
            if self.labels:
                labels = {item.id: self.labels[item.id] for item in items}
            return self.__class__(
                items=items,
                labels=labels,
            )
        return self.items[key]

    def get_items(self) -> List[BaseObject]:
        return list(self.items)

    def subset_by_ids(self, item_ids: List[str]) -> "DatasetBase":
        item_id_set = set(item_ids)
        items = [item for item in self.items if item.id in item_id_set]
        labels = None
        if self.labels:
            # safeguard against case where labels are not set
            labels = {
                item.id: self.labels.get(
                    item.id, Label(id=item.id, type=item.type, label="None")
                )
                for item in items
            }
        return DatasetBase(
            labels=labels,
            items=items,
        )

    def subset_by_ids_inplace(self, item_ids: List[str]) -> None:
        item_id_set = set(item_ids)
        self.items = [item for item in self.items if item.id in item_id_set]
        if self.labels:
            # safeguard against case where labels are not set
            self.labels = {
                item.id: self.labels.get(
                    item.id, Label(id=item.id, type=item.type, label="None")
                )
                for item in self.items
            }


class DataLoaderBase:
    @classmethod
    def load(cls, split_name: Optional[str] = None) -> DatasetBase:
        raise NotImplementedError()

    @classmethod
    def instantiate_and_load(cls, **kwargs) -> DatasetBase:
        split_name = None
        if "split_name" in kwargs:
            split_name = kwargs.pop("split_name")
        return cls(**kwargs).load(split_name=split_name)
