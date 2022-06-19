import torch
from torch.utils.data import DataLoader

from datasets.load import load_dataset
from datasets import Features, Array3D
from datasets.fingerprint import Hasher
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
from accelerate import Accelerator


pretrained = 'facebook/vit-mae-base'

feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained)


def preprocess(batch):
    return feature_extractor(batch['img'])


def main():
    # Resizing & Normalization
    dataset = load_dataset('cifar100', split='train')

    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        **dataset.features,
    })

    # Workaround for https://github.com/huggingface/datasets/issues/4521
    dataset = dataset.map(preprocess,
            features=features, new_fingerprint=Hasher.hash(f'{pretrained}-preprocess'),
            batched=True, batch_size=5000, writer_batch_size=5000)

    # Calculating embedding
    model = ViTMAEForPreTraining.from_pretrained(pretrained)

    dataset.set_format(type='torch', columns=['pixel_values'])
    data = DataLoader(dataset, batch_size=256)

    accelerator = Accelerator()
    model, data = accelerator.prepare(model, data)

    model.eval()
    with torch.no_grad():
        for inputs in data:
            outputs = model(output_hidden_states=True, **inputs)
            hidden_states = outputs.hidden_states
            print([hs.shape for hs in hidden_states])


if __name__ == '__main__':
    main()
