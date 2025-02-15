import random

from sklearn.datasets import make_blobs

from latent_spectrograms.visualize import Visualizer

try:
    from pywaspgen.burst_def import BurstDef
    from pywaspgen.iq_datagen import IQDatagen
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Could not find the required modules to run examples. Please install the necessary packages using 'pip install latent_spectrograms[examples].'"
    )


if __name__ == "__main__":
    # PyWASPGEN config path
    pywaspgen_cfg_path = "examples/pywaspgen_cfg.json"

    # Number of points to create
    n_samples = 25

    mod_classes = [
        {"format": "ask", "order": 2, "label": "BASK"},
        {"format": "pam", "order": 4, "label": "4PAM"},
        {"format": "qam", "order": 16, "label": "16QAM"},
    ]

    # Create synthetic 2D locations using make_blobs
    n_features = 2  # Assuming 2 features for visualization
    centers = len(mod_classes)  # Number of centers to generate

    # Generate random data
    data, labels = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers
    )

    def create_burst(burst_idx):
        return [
            BurstDef(
                cent_freq=random.uniform(-0.3, 0.3),
                bandwidth=random.uniform(0.2, 0.3),
                start=0,
                duration=1024,
                sig_type=mod_classes[burst_idx],
            )
        ]

    # Create synthetic IQ data using PyWASPGEN
    burst_lists = list(map(create_burst, labels))
    iq_data, burst_lists = IQDatagen(config_file=pywaspgen_cfg_path).gen_batch(
        burst_lists
    )
    text_labels = list(map(lambda x: mod_classes[x]["label"], labels))

    vis = Visualizer(
        x=data[:, 0],
        y=data[:, 1],
        text_labels=text_labels,
        iq_data=iq_data,
    )

    vis.visualize()
