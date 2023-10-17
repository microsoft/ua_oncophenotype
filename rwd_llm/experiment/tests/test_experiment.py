import os

from llm_lib.chains.mapping_chain import MappingChain
from llm_lib.data_loaders import (
    DataframeClinicalNoteDataLoader,
    DataframePatientDataLoader,
)
from llm_lib.eval import ClassificationEvaluation

from .. import DatasetRunner, Experiment

CUR_DIR = os.path.dirname(__file__)
SAMPLE_DATA_DIR = os.path.abspath(os.path.join(CUR_DIR, "../../tests/sample_data"))


def test_note_experiment():
    loader = DataframeClinicalNoteDataLoader(
        dataframe_path="sample_notes.tsv",
        text_column="text",
        date_column="date",
        patient_id_column="patient_id",
        note_id_column="note_id",
        type_column="type",
        note_type="Document.ClinicalNote",
        data_root_dir=SAMPLE_DATA_DIR,
        label_column="color",
    )
    # mapping from note id to label
    mapping = {
        "note_001": "red",
        "note_002": "red",
        "note_003": "red",
        "note_004": "yellow",
        "note_005": "yellow",
        "note_006": "yellow",
        "note_007": "green",
    }
    fake_chain = MappingChain(mapping=mapping, input_key="id")
    runner = DatasetRunner()
    evaluation = ClassificationEvaluation()

    experiment = Experiment(
        dataset=loader.load(),
        chain=fake_chain,
        data_runner=runner,
        evaluation=evaluation,
    )
    experiment.run()


def test_patient_experiment():
    loader = DataframePatientDataLoader(
        dataframe_path="sample_patient_labels.tsv",
        patient_id_column="patient_id",
        data_root_dir=SAMPLE_DATA_DIR,
        label_column="color_label",
    )
    # mapping from patient id to label
    mapping = {
        "pat_001": "red",
        "pat_002": "yellow",
        "pat_003": "green",
    }
    fake_chain = MappingChain(mapping=mapping, input_key="id")
    runner = DatasetRunner()
    evaluation = ClassificationEvaluation()

    experiment = Experiment(
        dataset=loader.load(),
        chain=fake_chain,
        data_runner=runner,
        evaluation=evaluation,
    )
    experiment.run()


if __name__ == "__main__":
    test_note_experiment()
    test_patient_experiment()
