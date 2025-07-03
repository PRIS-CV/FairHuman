import datasets
import json
import os
from os.path import join


DATA_DIR = os.environ.get("DATA_DIR")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text_english": datasets.Value("string"),
        "bounding_box": datasets.Value("string")
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default")


class CustomDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = join(DATA_DIR, "metadata.jsonl")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = [
                json.loads(line) for line in f
            ]

        for record in metadata:
            text_english = record["text_english"]

            image_path = record["image"]
            image_path = join(DATA_DIR, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = record["conditioning_image"]
            conditioning_image_path = join(DATA_DIR, conditioning_image_path)
            conditioning_image = open(conditioning_image_path, "rb").read()

            mask_hand_image_path = record["mask_hand"]
            mask_hand_image_path = join(DATA_DIR, mask_hand_image_path)
            mask_hand_image = open(mask_hand_image_path, "rb").read()

            mask_face_image_path = record["mask_face"]
            mask_face_image_path = join(DATA_DIR, mask_face_image_path)
            mask_face_image = open(mask_face_image_path, "rb").read()


            yield record["image"], {
                "text_english": text_english,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
                "mask_hand": {
                    "path": mask_hand_image_path,
                    "bytes": mask_hand_image,
                },
                "mask_face": {
                    "path": mask_face_image_path,
                    "bytes": mask_face_image,
                },
                
            }