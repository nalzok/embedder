import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader

from datasets.load import load_dataset
from datasets import Features, Array3D
from datasets.fingerprint import Hasher
from transformers import AutoFeatureExtractor, ViTMAEModel


def main():
    dataset = load_dataset('cifar100', split='train')
    pretrained = 'facebook/vit-mae-base'

    device = xm.xla_device()

    # Resizing & Normalization
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained)

    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        **dataset.features,
    })

    # Workaround for https://github.com/huggingface/datasets/issues/4521
    dataset = dataset.map(lambda examples: feature_extractor(examples['img']),
            features=features, new_fingerprint=Hasher.hash(f'{pretrained}-preprocess'),
            batched=True, batch_size=5000, writer_batch_size=5000)

    # Calculating embedding
    model = ViTMAEModel.from_pretrained(pretrained)
    model = model.to(device)

    dataset.set_format(type='torch', columns=['pixel_values'])
    data = DataLoader(dataset, batch_size=256, num_workers=4)

    model.eval()
    with torch.no_grad():
        for inputs in data:
            pixel_values = inputs['pixel_values'].to(device)
            outputs = model(pixel_values)
            last_hidden_states = outputs.last_hidden_state[:, 0, :]    # only look at the [CLS] token
            print(last_hidden_states.shape, last_hidden_states[0, 0])


if __name__ == '__main__':
    # torch_xla._XLAC._xla_set_use_full_mat_mul_precision(True)
    main()
