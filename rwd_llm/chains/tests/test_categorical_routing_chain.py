"""
Example based on the following decision graph:

legs? |--> 0 -> breathes air? |--> No  -> Fish
      |                       |--> Yes -> Snake
      |
      |--> 2 -> can fly? |--> No  -> Human
      |                  |--> Yes -> Bird
      |
      |--> 4 -> has hooves? |--> No  -> climbs trees? |--> No  -> Dog
      |                     |                         |--> Yes -> Cat
      |                     |
      |                     |--> Yes -> tastes good? |--> No  -> Horse
      |                                              |--> Yes -> Pig
      |
      |--> 6 -> makes honey? |--> No -> Ant
                             |--> Yes -> Bee
"""
import logging
import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from langchain.chains.base import Chain

from ..categorical_routing_chain import CategoricalRoutingChain, ChainNode
from ..mapping_chain import MappingChain
from ..no_op_chain import NoOpChain

CUR_DIR = os.path.dirname(__file__)


def _run_tests(chain: CategoricalRoutingChain):
    input = dict(
        legs="0",
        can_fly="No",
        breathes_air="No",
        makes_honey="No",
        tastes_good="Yes",
        climbs_trees="No",
        has_hooves="No",
    )
    result = chain(input)
    assert result["label"] == "Fish"

    input = dict(
        legs="2",
        can_fly="No",
        breathes_air="Yes",
        makes_honey="No",
        tastes_good="Ewwww",
        climbs_trees="Yes",
        has_hooves="No",
    )
    result = chain(input)
    assert result["label"] == "Human"

    input = dict(
        legs="4",
        can_fly="No",
        breathes_air="Yes",
        makes_honey="No",
        tastes_good="No!!",
        climbs_trees="Yes",
        has_hooves="No",
    )
    result = chain(input)
    assert result["label"] == "Cat"

    input = dict(
        legs="4",
        can_fly="No",
        breathes_air="Yes",
        makes_honey="No",
        tastes_good="Yes",
        climbs_trees="Yes",
        has_hooves="Yes",
    )
    result = chain(input)
    assert result["label"] == "Pig"

    input = dict(
        legs="6",
        can_fly="No",
        breathes_air="Yes",
        makes_honey="No",
        tastes_good="No",
        climbs_trees="Yes",
        has_hooves="No",
    )
    result = chain(input)
    assert result["label"] == "Ant"


def test_normal_usage():
    # Fish, Snake, Ant, Bee, Human, Bird, Cat, Dog, Horse, Pig
    legs = NoOpChain(expected_keys=["legs"], the_key="legs")
    has_hooves = NoOpChain(expected_keys=["has_hooves"], the_key="has_hooves")
    breathes_air = MappingChain(
        input_key="breathes_air", mapping={"No": "Fish", "Yes": "Snake"}
    )
    can_fly = MappingChain(input_key="can_fly", mapping={"No": "Human", "Yes": "Bird"})
    makes_honey = MappingChain(
        input_key="makes_honey", mapping={"No": "Ant", "Yes": "Bee"}
    )
    climbs_trees = MappingChain(
        input_key="climbs_trees", mapping={"No": "Dog", "Yes": "Cat"}
    )
    tastes_good = MappingChain(
        input_key="tastes_good", mapping={"No": "Horse", "Yes": "Pig"}
    )

    input_keys = [
        "legs",
        "has_hooves",
        "breathes_air",
        "can_fly",
        "makes_honey",
        "climbs_trees",
        "tastes_good",
    ]

    nodes = [
        ChainNode(
            chain=legs,
            name="legs",
            children={
                "0": "breathes_air",
                "2": "can_fly",
                "4": "has_hooves",
                "6": "makes_honey",
            },
        ),
        ChainNode(
            chain=has_hooves,
            name="has_hooves",
            children={"No": "climbs_trees", "Yes": "tastes_good"},
        ),
        ChainNode(chain=breathes_air, name="breathes_air"),
        ChainNode(chain=can_fly, name="can_fly"),
        ChainNode(chain=makes_honey, name="makes_honey"),
        ChainNode(chain=climbs_trees, name="climbs_trees"),
        ChainNode(chain=tastes_good, name="tastes_good"),
    ]

    chain = CategoricalRoutingChain.from_node_list(
        node_list=nodes, input_keys=input_keys, output_keys=["label"]
    )

    _run_tests(chain)


def test_from_config():
    initialize_config_dir(
        config_dir=CUR_DIR, job_name="simple_configs_test", version_base="1.3"
    )
    # load config from categorical_routing_chain_config.yaml
    cfg = compose(config_name="categorical_routing_chain_config")
    chain: CategoricalRoutingChain = instantiate(cfg.chain, _convert_="partial")
    _run_tests(chain)
    GlobalHydra.instance().clear()


def test_memory():
    # this is just testing that the outputs of each chain are stored for later use
    first_node = MappingChain(
        input_key="first_input", output_key="first_output", default="foo"
    )
    second_node = MappingChain(
        input_key="first_output",
        output_key="second_output",
        mapping={"foo": "bar"},
        default="ERROR",
    )
    third_node = MappingChain(
        input_key="first_output",
        output_key="third_output",
        mapping={"foo": "baz"},
        default="ERROR",
    )
    nodes = [
        ChainNode(
            chain=first_node,
            name="first_node",
            children={
                "": "second_node",
            },
        ),
        ChainNode(
            chain=second_node,
            name="second_node",
            children={"": "third_node"},
        ),
        ChainNode(chain=third_node, name="third_node"),
    ]

    chain = CategoricalRoutingChain.from_node_list(
        node_list=nodes,
        input_keys=["first_input"],
        output_keys=["third_output"],
        memorized_keys=["first_output"],
    )
    output = chain({"first_input": "some input"})
    # if we can get the correct third output, it means the first output was memorized
    assert output["third_output"] == "baz"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_normal_usage()
    test_from_config()
    test_memory()
