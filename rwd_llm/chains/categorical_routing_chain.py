import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.pydantic_v1 import BaseModel, validator

from ..memory import EphemeralMemoryProvider, PersistentMemoryProviderBase

logger = logging.getLogger(__name__)

DEFAULT_LABEL = "<default>"


class ChainNode(BaseModel):
    chain: Chain
    name: str
    # Mapping from label to next chain (by name). If no mapping, this is a leaf node.
    children: Optional[Dict[str, str]] = None
    # Key where the label is stored in the 'chain' output.
    label_output_key: Optional[str] = None
    # Key to use to pass the label to the next chain; None means don't pass it.
    label_next_input_key: Optional[str] = None

    @validator("children")
    def _validate_children(cls, children):
        if not children:
            return None
        default_chain = None
        for label, next_chain in children.items():
            if not label or label == DEFAULT_LABEL:
                # use this as default
                if default_chain is not None and default_chain != next_chain:
                    raise ValueError(
                        f"Multiple defaults: {default_chain}, {next_chain}"
                    )
                default_chain = next_chain
        if default_chain is not None:
            children[DEFAULT_LABEL] = default_chain
        return children

    @validator("label_output_key", always=True)
    def _validate_label_output_key(cls, label_output_key, values):
        if label_output_key is None:
            chain: Chain = values["chain"]
            chain_output_keys = chain.output_keys
            label_output_key = chain_output_keys[0]
            if len(chain_output_keys) > 1:
                if values.get("children"):
                    # this isn't a leaf node, so should probably have label_output_key
                    # explicitly set
                    logger.warning(
                        f"Chain {values['name']} has multiple output keys. "
                        f"Using {chain_output_keys[0]} as the label output key."
                    )
        return label_output_key

    @property
    def label_key(self) -> str:
        if self.label_output_key is None:
            raise ValueError(
                "No label_output_key, this should have been set by validator"
            )
        return self.label_output_key


class CategoricalRoutingChain(Chain):
    """Chain that wraps a graph of chains and routes to the correct one based on the
    output of the previous chain."""

    root: ChainNode
    # mapping from names to chains
    nodes: Dict[str, ChainNode]
    # mapping from chain (by name) to mapping from label to next chain (by name)
    edges: Dict[str, Dict[str, str]]
    # not sure what leaf nodes will return, so have to specify
    return_keys: List[str]
    persistent_memory_provider: PersistentMemoryProviderBase
    # if None, will be whatever the root chain expects
    expected_keys: Optional[List[str]] = None
    # if intermediate results should be memorized for use by downstream nodes, specify
    # them here
    memorized_keys: Optional[List[str]] = None
    item_id_key: Optional[str] = None

    @validator("persistent_memory_provider", pre=True)
    def _validate_persistent_memory_provider(cls, persistent_memory_provider):
        logger.info(
            "Validation: persistent_memory_provider is"
            f" {type(persistent_memory_provider)}"
        )
        if persistent_memory_provider is None:
            return EphemeralMemoryProvider()
        return persistent_memory_provider

    @classmethod
    def from_node_list(
        cls,
        node_list: List[ChainNode],
        output_keys: List[str],
        persistent_memory_provider: Optional[PersistentMemoryProviderBase] = None,
        input_keys: Optional[List[str]] = None,
        # memory: Optional[BaseMemory] = None,
        memorized_keys: Optional[List[str]] = None,
        item_id_key: Optional[str] = None,
    ) -> "CategoricalRoutingChain":
        nodes = {node.name: node for node in node_list}
        edges = {node.name: node.children for node in node_list if node.children}
        persistent_memory_provider = (
            persistent_memory_provider or EphemeralMemoryProvider()
        )
        return cls(
            root=node_list[0],
            nodes=nodes,
            edges=edges,
            expected_keys=input_keys,
            return_keys=output_keys,
            # memory=memory,
            memorized_keys=memorized_keys,
            persistent_memory_provider=persistent_memory_provider,
            item_id_key=item_id_key,
        )

    @staticmethod
    def sequential_node_list(chains: List[Chain]) -> List[ChainNode]:
        names = [
            f"chain_{chain_idx}_{type(chain).__name__}"
            for chain_idx, chain in enumerate(chains)
        ]
        nodes_list = []
        for chain_idx, chain in enumerate(chains):
            name = names[chain_idx]
            children = None
            if chain_idx < len(chains) - 1:
                children = {DEFAULT_LABEL: names[chain_idx + 1]}
            nodes_list.append(
                ChainNode(
                    chain=chain,
                    name=name,
                    children=children,
                    label_output_key=None,
                )
            )
        return nodes_list

    @property
    def input_keys(self) -> List[str]:
        """Either user-specified, or whatever keys the root prompt expects."""
        return self.expected_keys or self.root.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Expect output keys"""
        return self.return_keys

    def _prep_inputs(
        self,
        chain_node: ChainNode,
        item_id: str,
        inputs: Dict[str, Any],
        orig_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get expected inputs either from current inputs (from previous node), from
        original chain inputs, or from memory"""
        prepped_inputs: Dict[str, Any] = {}
        for key in chain_node.chain.input_keys:
            if key in inputs:
                prepped_inputs[key] = inputs[key]
            elif key in self.persistent_memory_provider.keys(item_id=item_id):
                logger.debug(
                    f"getting missing input key {key} for chain"
                    f" {chain_node.name} from memory provider"
                )
                prepped_inputs[key] = self.persistent_memory_provider.get_memory(
                    item_id=item_id, key=key
                )
            elif key in orig_inputs:
                logger.debug(
                    f"getting missing input key {key} for chain"
                    f" {chain_node.name} from original inputs"
                )
                prepped_inputs[key] = orig_inputs[key]
            else:
                # it's ok, we may get this input from the chain's memory
                logger.debug(
                    f"Missing input key {key} for chain"
                    " not found, hopefully we'll get it from the chain's memory"
                )
        return prepped_inputs

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        cur_node = self.root

        if self.item_id_key:
            item_id = inputs[self.item_id_key]
        else:
            # Just set a random ID for this run. Only useful if we don't actually care
            # about persisting results.
            item_id = uuid.uuid4().hex
        logger.info(f"Item ID: {item_id}")  # REMOVE ME
        logger.info(
            f"Memory Provider: {type(self.persistent_memory_provider)}"
        )  # REMOVE ME

        orig_inputs = dict(**inputs)
        keys_to_memorize = set(self.output_keys).union(self.memorized_keys or [])
        logger.debug(f"will memorize keys: {keys_to_memorize}")
        inputs = {}
        while True:
            # logger.debug(f"Outputs of chain node {cur_node.name}: {outputs}")
            # memorize outputs that match output keys
            keys_to_save = {
                key for key in cur_node.chain.output_keys if key in keys_to_memorize
            }
            try:
                chain_inputs = self._prep_inputs(
                    cur_node, item_id=item_id, inputs=inputs, orig_inputs=orig_inputs
                )
                logger.info(f"Running chain node {cur_node.name}")
                callbacks = _run_manager.get_child()
                outputs = cur_node.chain(
                    chain_inputs, callbacks=callbacks, return_only_outputs=True
                )
            except Exception as e:
                self.persistent_memory_provider.log_error(
                    item_id=item_id, chain_name=cur_node.name, error=e
                )
                raise e
            # logger.debug(f"Outputs of chain node {cur_node.name}: {outputs}")
            # memorize outputs that match output keys
            to_save = {key: outputs[key] for key in keys_to_save}
            # print(f"  chain output keys: {outputs.keys()}")
            # print(f"  keys to memorize: {keys_to_memorize}")
            # print(f"Saving keys {to_save.keys()}")
            self.persistent_memory_provider.add_memories(
                item_id=item_id, memories=to_save
            )
            if cur_node.name not in self.edges:
                # this is a leaf node, return the result
                break
            label = outputs[cur_node.label_key]
            # mapping from label value to next node
            output_mapping = self.edges[cur_node.name]
            # set the inputs for the next node
            inputs = dict(**outputs)
            if cur_node.label_next_input_key:
                # rename the output value for the next input
                val = inputs.pop(cur_node.label_key)
                inputs[cur_node.label_next_input_key] = val
            # find the next node
            if str(label) in output_mapping:
                cur_node = self.nodes[output_mapping[label]]
            elif DEFAULT_LABEL in output_mapping:
                cur_node = self.nodes[output_mapping[DEFAULT_LABEL]]
            else:
                raise ValueError(
                    f"Label {label} not found in output mapping for chain node"
                    f" {cur_node.name}: {output_mapping}"
                )
        outputs = {
            key: self.persistent_memory_provider.get_memory(item_id=item_id, key=key)
            for key in self.output_keys
        }

        return outputs
