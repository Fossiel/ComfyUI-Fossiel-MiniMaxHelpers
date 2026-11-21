from .mmhic import FossielMMHImageConditioning
from .phic import FossielPHImageCond
from .rtvp import FossielMiniMaxRefToVid
from .rtvpn import FossielMiniMaxRefToVidNat

NODE_CLASS_MAPPINGS = {
    "FossielMMHImageConditioning": FossielMMHImageConditioning,
    "FossielPHImageCond": FossielPHImageCond,
    "FossielMiniMaxRefToVid": FossielMiniMaxRefToVid,
    "FossielMiniMaxRefToVidNat": FossielMiniMaxRefToVidNat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FossielMMHImageConditioning": "MiniMax H3 Image Conditioning - Natural",
    "FossielPHImageCond": "P-H3 Image Conditioning - Natural",
    "FossielMiniMaxRefToVid": "MiniMax H3 Reference to Video (Perma-Ports)",
    "FossielMiniMaxRefToVidNat": "MiniMax H3 Reference to Video - Natural"
}
