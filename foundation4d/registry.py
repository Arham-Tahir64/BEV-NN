"""Viewer factory: explicit new-model dispatch; legacy state keys stay intact."""
def build_model(config):
    if config.get('architecture')=='foundation4d_lite':
        from .runtime import build_model as build_lite
        return build_lite(config)
    from src.models.occupancy_model import OccupancyModel
    return OccupancyModel(**{**config,'pretrained':False})
