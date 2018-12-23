import logging
import chainer
import chainercv
import sklearn.preprocessing
import numpy as np
import pandas as pd
import albumentations as alb
import joblib
import json
import six
import cv2
import os
from tqdm import tqdm
from . import utils

log = logging.getLogger(__name__)

cache = joblib.Memory(os.path.join(os.environ['HOME'], '.cache'), verbose=2)


class OpenImagesLabels(object):
    """There are 3 ways to express a label:

        - the readable label: Vehicle

        - the label id: /m0123azdb

        - the encoded label: 42
    """

    def __init__(self, label_df):
        self.label_df = label_df
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.label_encoder.fit(self.label_df.index.unique())

    def readable_label_of_label_id(self, label_id):
        return self.label_df.loc[label_id].readable_label

    def readable_label_of_encoded_label(self, encoded_label):
        return self.readable_label_of_label_id(
            self.label_encoder.inverse_transform(encoded_label))

    def encoded_label_of_label_id(self, label_id):
        return self.label_encoder.transform(label_id)

    def label_ids(self):
        return self.label_df.index

    def readable_classes(self):
        return [
            self.readable_label_of_label_id(x)
            for x in self.label_encoder.classes_
        ]

    def num_classes(self):
        return len(self.label_encoder.classes_)


def openimages_label_encoder(root):
    log.info("openimages: creating label encoder")
    label_df = pd.read_csv(
        os.path.join(root, "class-descriptions-boxable.csv"),
        header=None,
        names=['label_id', 'readable_label']).set_index('label_id')

    label_manager = OpenImagesLabels(label_df)
    return label_manager


class TransformLabeled(object):
    def __init__(self, f, **kwargs):
        self.f = f
        self.kwargs = kwargs

    def __call__(self, x):
        image, label = x
        image = self.f(image, **self.kwargs)
        return image, label


def image_to_color(image):
    if len(image) == 2:
        image, label = image
    if image.shape[0] == 1:
        return np.concatenate((image, image, image), axis=0)
    elif image.shape[0] == 4:
        return image[[0, 1, 2], :, :]
    else:
        return image


resize_warning_sent = False


def resize_if_needed(image, size):
    global resize_warning_sent
    if image.shape[1:] != size:
        if not resize_warning_sent:
            log.warning("resizing image from %s to %s", image.shape[1:], size)
            resize_warning_sent = True
        return chainercv.transforms.resize_contain(image, size=size)
    else:
        return image


class AlbLabeledImageDataset(chainer.dataset.dataset_mixin.DatasetMixin):
    """Like chainer.datasets.LabeledImageDataset, but uses cv2 instead
    of PIL (2.5x faster at loading 224x224 jpegs), and accepts an
    albumentations augmentation (which needs to work on HWC RGB rather
    than the CHW RGB that chainer consumes).
    """

    def __init__(self,
                 pairs,
                 root='.',
                 dtype=np.float32,
                 label_dtype=np.int32,
                 augmentation=None):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype
        self._augmentation = augmentation

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = cv2.imread(full_path, cv2.IMREAD_COLOR)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        if self._augmentation is not None:
            image = self._augmentation(image=image)['image']

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(self._dtype)

        # This should hold given that we pass IMREAD_COLOR to imread().
        assert image.ndim == 3, image.shape

        label = np.array(int_label, dtype=self._label_dtype)
        return image, label


class OpenImagesLabelCleaner(object):
    def __init__(self, root, label_ids):
        json_file = os.path.join(root, "bbox_labels_600_hierarchy.json")
        log.debug("loading OpenImages label hierarchy: %s", json_file)
        with open(json_file, 'r') as f:
            hier = json.load(f)
        parents_dict = {}
        self._build_parents(hier, parents_dict, [])
        # log.debug("parents: %s", parents_dict)
        self.parents = self._df_of_dict(parents_dict)
        self.parents = self.parents[self.parents.ParentLabelName.isin(
            set(label_ids))]
        # log.debug("parents df:\n%s", self.parents)

    def _df_of_dict(self, parents_dict):
        return pd.DataFrame(
            [(k, v) for k, vv in parents_dict.items() for v in vv],
            columns=["ChildLabel", "ParentLabelName"]).set_index("ChildLabel")

    def _build_parents(self, hier, label_parents, current_parents):
        if "LabelName" not in hier:
            return
        label = hier["LabelName"]
        # assert label not in label_parents, (label, label_parents)
        # Do not append(). current_parents should be a new object.
        current_parents = current_parents + [label]
        # We do want a given label to be included in its own parents.
        label_parents[label] = current_parents
        for sub in hier.get("Subcategory", []):
            self._build_parents(sub, label_parents, current_parents)

    def clean(self, annotations):
        log.debug("joining\n%s\n with\n%s", annotations.head(1),
                  self.parents.head(1))
        return annotations.join(
            self.parents,
            on="LabelName")[["ImageID",
                             "ParentLabelName"]].drop_duplicates().rename(
                                 columns=dict(ParentLabelName="LabelName"))


def debug_annotations(annot, images, label_encoder):
    images = set(images)
    annot = annot[annot.ImageID.isin(images)].copy()
    log.debug("annot labelname:\n%s", annot.LabelName)
    unknown_labels = set(annot.LabelName).difference(label_encoder.label_ids())
    if unknown_labels:
        log.warning(
            "some labels in the annotation/hierarchy" +
            " are unknown to the encoder: %s", unknown_labels)
    annot['DebugLabel'] = [
        label_encoder.readable_label_of_label_id(x) for x in annot.LabelName
    ]
    return annot


@cache.cache
def openimages_dataset(root,
                       subset='train',
                       image_glob='*.jpg',
                       size=(224, 224),
                       label_encoder=None,
                       augmentation=None,
                       label_zero=0.,
                       label_one=1.,
                       version=5):
    """version arg is solely for making sure the cache is not reused
    if the implementation changes.
    """
    log.info("openimages: creating dataset for subset %s", subset)
    if label_encoder is None:
        label_encoder = openimages_label_encoder(root)

    log.info("openimages: reading image annotations")
    image_annotations = pd.read_csv(
        os.path.join(
            root,
            "{}-annotations-human-imagelabels-boxable.csv".format(subset)))
    image_annotations = image_annotations[image_annotations.Confidence > 0.8]
    image_annotations = image_annotations[['ImageID', 'LabelName']]

    log.info("before cleaning, %d annotations", len(image_annotations.index))
    debug_images = image_annotations.ImageID.iloc[:10]
    log.debug(
        "image annotations before cleaning:\n%s",
        debug_annotations(image_annotations, debug_images, label_encoder))

    log.info("openimages: cleaning labels")
    label_cleaner = OpenImagesLabelCleaner(root, label_encoder.label_ids())
    image_annotations = label_cleaner.clean(image_annotations)

    log.info("after cleaning, %d annotations", len(image_annotations.index))
    log.debug("clean returns:\n%s", image_annotations.head(5))
    log.debug(
        "image annotations after cleaning:\n%s",
        debug_annotations(image_annotations, debug_images, label_encoder))

    def file_name_of_image_id(image_id):
        return root + "/" + subset + "/" + image_id + ".jpg"

    log.info("openimages: computing image labels")

    image_annotations['LabelName'] = label_encoder.encoded_label_of_label_id(
        image_annotations.LabelName)
    image_annotations['image_file'] = image_annotations.ImageID + ".jpg"
    image_annotations['image_path'] = (
        root + "/" + subset + "/" + image_annotations.image_file)

    all_files = set(os.listdir(os.path.join(root, subset)))

    image_annotations = image_annotations[image_annotations.image_file.isin(
        all_files)]
    image_annotations = image_annotations[['image_path', 'LabelName']]

    num_classes = label_encoder.num_classes()

    def process_group(ig):
        image_path, group = ig
        labels = group.LabelName.values
        label_vector = np.zeros(num_classes, dtype=np.float32) + label_zero
        label_vector[labels] = label_one
        return (image_path, label_vector)

    # a parallel version with joblib.Paralell is 1000x slower!
    pairs = [
        process_group(g)
        for g in tqdm(image_annotations.groupby('image_path'))
    ]

    log.info("openimages: dataset: creating labeled dataset")
    dataset = AlbLabeledImageDataset(
        pairs=pairs,
        root=root,
        label_dtype=np.float32,
        augmentation=augmentation)
    dataset = chainer.datasets.TransformDataset(
        dataset, TransformLabeled(resize_if_needed, size=size))
    # dataset = chainer.datasets.TransformDataset(
    #     dataset, TransformLabeled(image_to_color))
    dataset.files = [x[0] for x in pairs]

    return dataset


def classification_augmentation():
    """Rationalized version of the one above.
    """

    return alb.Compose(
        [
            alb.HorizontalFlip(p=.5),  # 128 mus
            alb.OneOf(
                [
                    # These two do the same thing I think. Keeping only
                    # the faster one.
                    alb.IAAAdditiveGaussianNoise(p=1.),  # 484 mus
                    # alb.GaussNoise(p=1.),  # 1.11 ms
                ],
                p=0.2),  # 1.03 ms with both
            alb.OneOf([
                alb.MotionBlur(p=1.),
                alb.MedianBlur(blur_limit=3, p=1.),
                alb.Blur(blur_limit=3, p=1.),
            ],
                      p=0.2),  # 40 mus
            alb.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.8),
            # (above) 348 mus
            alb.OneOf(
                [
                    alb.OpticalDistortion(p=1.),  # 95 mus
                    alb.GridDistortion(p=1.),  # 101 mus
                    # alb.IAAPiecewiseAffine(p=1.),  # 5.61 ms
                ],
                p=0.2),  # 2.48 ms -> 113 mus with the 2 first ones
            alb.OneOf([
                alb.CLAHE(clip_limit=2, p=1.),
                alb.IAASharpen(p=1.),
                alb.IAAEmboss(p=1.),
                alb.RandomContrast(p=1.),
                alb.RandomBrightness(p=1.),
            ],
                      p=0.3),  # 257 mus
            alb.HueSaturationValue(p=0.3),  # 395 mus
        ],
        p=0.9)  # 3.84 ms -> 1.52 ms


def transform_labeled(aug):
    def f(x_label):
        x, label = x_label
        x = np.transpose(x, (1, 2, 0)).astype(np.uint8)
        x = aug(image=x)['image']
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        return x, label

    return f


def load_openimages_datasets(root='/media/rlehy/datasets/openimages-v4/bb'):
    datasets = utils.AttrDict()
    label_encoder = openimages_label_encoder(root)

    datasets.train = openimages_dataset(
        root,
        subset='train',
        label_encoder=label_encoder,
        augmentation=classification_augmentation(),
        label_zero=0.1,
        label_one=0.99)
    datasets.validation = openimages_dataset(
        root, subset='validation', label_encoder=label_encoder,
        label_zero=0.1,
        label_one=0.99)

    datasets.label_names = label_encoder.readable_classes()
    datasets.string_of_label = label_encoder.readable_label_of_encoded_label
    datasets.id_name = 'ImageID'
    return datasets
