from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"
    
    # Custom dataset options for SMILES dataset
    # Path to CSV file for custom dataset format
    cfg.dataset.csv_path = 'NHC-cracker-zzy-v1.csv'
    
    # Whether to use dE_triplet as graph-level input feature instead of prediction target
    # If True: dE_triplet is stored in data.u and excluded from prediction targets
    # If False: dE_triplet is included in prediction targets (default behavior)
    cfg.dataset.use_dE_triplet_as_feature = False