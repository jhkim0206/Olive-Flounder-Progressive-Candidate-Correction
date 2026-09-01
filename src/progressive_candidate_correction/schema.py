"""Class and channel order used throughout the study."""

SYMPTOM_CLASS_NAMES = (
    "body_lesion",
    "mouth_ulcer",
    "fin_deformity",
    "fin_necrosis",
    "fin_base_necrosis",
    "caudal_deformity",
    "caudal_necrosis",
    "caudal_base_necrosis",
)

SEMANTIC_CLASS_NAMES = ("background", *SYMPTOM_CLASS_NAMES)

PART_MAP_CLASS_NAMES = ("background", "body", "fin", "caudal_fin", "mouth")

ROUTE_NAMES = ("body", "mouth", "fin", "caudal_fin")
PART_MAP_ROUTE_INDICES = (1, 4, 2, 3)
PART_MAP_TO_ROUTE = (0, 0, 2, 3, 1)

VISUAL_EVIDENCE_NAMES = ("redness", "shape", "lesion", "unaffected_surface")

ZONE_NAMES = (
    "body",
    "mouth",
    "fin_tip",
    "fin_middle",
    "fin_base",
    "caudal_fin_tip",
    "caudal_fin_middle",
    "caudal_fin_base",
)

ZONE_LABEL_NAMES = ("background", *ZONE_NAMES)

# Semantic labels use background route 0; symptom labels follow ROUTE_NAMES.
SEMANTIC_TO_ROUTE = (0, 0, 1, 2, 2, 2, 3, 3, 3)

NUM_SEMANTIC_CLASSES = len(SEMANTIC_CLASS_NAMES)
NUM_SYMPTOM_CLASSES = len(SYMPTOM_CLASS_NAMES)
NUM_PART_CLASSES = len(PART_MAP_CLASS_NAMES)
