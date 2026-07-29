import os
import datasets
import pandas as pd

_CITATION = """\
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
"""

_DATASETNAME = "med_qa"
_DISPLAYNAME = "MedQA"

_DESCRIPTION = """\
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

_HOMEPAGE = "https://github.com/jind11/MedQA"

_LICENSE = 'UNKNOWN'

_URL = "medqa.zip"

task_list = [
    "basic_medicine"    
]


class MedqaConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Medqa(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MedqaConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {               
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "E": datasets.Value("string"),
                "answer": datasets.Value("string")                
            }
        )
        return datasets.DatasetInfo(description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,)

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test", f"{task_name}_test.csv"),
                },
            ),
            #datasets.SplitGenerator(
            #    name=datasets.Split.VALIDATION,
             #   gen_kwargs={
            #        "filepath": os.path.join(data_dir, "val", f"{task_name}_val.csv"),
             #   },
           # ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev", f"{task_name}_dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, encoding="utf-8")
        df.columns = ["question", "A", "B", "C", "D","E", "answer"]
 
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance