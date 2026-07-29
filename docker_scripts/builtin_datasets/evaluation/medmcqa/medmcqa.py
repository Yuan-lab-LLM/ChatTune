import os
import datasets
import pandas as pd

task_list = [
    "medical"    
]

_URL="medmcqa.zip"

class MedmcqaConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Medmcqa(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MedmcqaConfig(
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
                "answer": datasets.Value("string")                
            }
        )
        return datasets.DatasetInfo(features=features,)

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
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev", f"{task_name}_dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, encoding="utf-8")
        df.columns = ["question", "A", "B", "C", "D", "answer"]
 
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance
