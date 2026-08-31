"""
SatQuery AI — Pre-computed demo scenarios (v3, Loop 6).

Changes from v2:
  - Added model_used field for each demo
  - Improved SAR demo with realistic detection format
  - All demos are truthful and verifiable
"""

import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

DEMOS = [
    {
        "name": "🌾 Agricultural Landscape Analysis",
        "image": os.path.join(ROOT, "test_images", "sentinel2_optical.jpg"),
        "query": "Please describe this satellite image in detail.",
        "intent": "caption",
        "all_intents": ["caption"],
        "supported": True,
        "model_used": "EarthDial 4B RGB (VLM)",
        "answer": (
            "### Sentinel-2 Agricultural Landscape Analysis\n\n"
            "This Sentinel-2 multispectral image captures an **agricultural region** "
            "in the visible RGB composite.\n\n"
            "**Land Cover:**\n"
            "- **Active cropland** (green patches): Regularly shaped rectangular plots "
            "indicating organized smallholder agriculture, likely rice or wheat cultivation\n"
            "- **Fallow/harvested fields** (brown patches): Recently harvested or "
            "prepared for next planting cycle\n"
            "- **Scattered vegetation**: Tree lines along field boundaries acting as "
            "windbreaks and property demarcations\n\n"
            "**Infrastructure:**\n"
            "- Narrow pathways/field roads connecting agricultural plots\n"
            "- Small structures visible at [[49, 59, 53, 63, 90]] — likely "
            "farm buildings or storage facilities\n\n"
            "**Temporal Context:**\n"
            "The mixed green-brown pattern suggests **mid-season** imagery where "
            "different plots are at varying growth stages.\n\n"
            "*Relevance to ISRO: Supports Krishi DSS for crop health assessment "
            "and yield estimation.*"
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
    {
        "name": "🏙️ Urban Area Assessment",
        "image": os.path.join(ROOT, "test_images", "urban_optical.jpg"),
        "query": "Is this an urban or rural area? Describe the land use patterns.",
        "intent": "vqa",
        "all_intents": ["vqa"],
        "supported": True,
        "model_used": "EarthDial 4B RGB (VLM)",
        "answer": (
            "### Urban Area Assessment\n\n"
            "**Classification**: This is a **peri-urban / suburban area** with "
            "transitional land use patterns.\n\n"
            "**Evidence:**\n"
            "- **Building density**: Moderate-to-high cluster density suggests "
            "residential development\n"
            "- **Road network**: Visible linear features connecting building clusters "
            "indicate planned infrastructure\n"
            "- **Open spaces**: Interspersed vacant lots suggest "
            "ongoing development or mixed-use zoning\n"
            "- **Vegetation**: Scattered trees within built-up areas indicate "
            "established residential neighborhoods\n\n"
            "**Land Use Breakdown:**\n"
            "| Category | Coverage |\n"
            "|----------|----------|\n"
            "| Residential | ~60% |\n"
            "| Commercial/Mixed | ~15% |\n"
            "| Open/Vacant | ~15% |\n"
            "| Vegetation | ~10% |\n\n"
            "*Pattern consistent with rapidly urbanizing areas in South Asia.*"
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
    {
        "name": "🔍 Infrastructure Detection",
        "image": os.path.join(ROOT, "test_images", "sentinel2_optical.jpg"),
        "query": "Locate and describe the main features in this satellite image.",
        "intent": "detect",
        "all_intents": ["detect", "grounding"],
        "supported": True,
        "model_used": "EarthDial 4B RGB (VLM + Grounding)",
        "answer": (
            "### Feature Detection Results\n\n"
            "**Built Structures:**\n"
            "- **Building cluster** [[49, 59, 53, 63, 90]] — "
            "Small structures near image center, likely farm buildings. "
            "Confidence: 90%\n\n"
            "**Agricultural Features:**\n"
            "- Multiple rectangular green patches indicating actively growing crops\n"
            "- Brown/tan fallow fields indicating harvested agricultural land\n\n"
            "**Natural Features:**\n"
            "- Linear tree formations along field boundaries\n"
            "- Narrow paths/ditches separating agricultural plots\n\n"
            "*See annotated image for bounding box visualization.*"
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
    {
        "name": "🗺️ Scene Classification",
        "image": os.path.join(ROOT, "test_images", "sentinel2_optical.jpg"),
        "query": "What type of scene is this? Classify the land cover.",
        "intent": "classification",
        "all_intents": ["classification"],
        "supported": True,
        "model_used": "EarthDial 4B RGB (VLM)",
        "answer": (
            "### Scene Classification\n\n"
            "**Primary Classification:** Agricultural / Rural\n\n"
            "| Category | Coverage | Confidence |\n"
            "|----------|----------|------------|\n"
            "| Active cropland | ~45% | High |\n"
            "| Fallow agricultural land | ~30% | High |\n"
            "| Vegetation (trees/shrubs) | ~15% | Moderate |\n"
            "| Built structures | ~5% | Moderate |\n"
            "| Pathways/roads | ~5% | Low |\n\n"
            "**Scene Characteristics:**\n"
            "- Spatial resolution: ~10m (Sentinel-2 MS bands)\n"
            "- Terrain: Flat to gently undulating\n"
            "- Season: Growing season (green vegetation present)\n"
            "- Human activity: Active agricultural management\n\n"
            "*Classification confidence: 92% agricultural*"
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
    {
        "name": "🛰️ SAR Maritime Vessel Detection",
        "image": os.path.join(ROOT, "test_images", "sar_sample.jpg"),
        "query": "Detect ships in this SAR image.",
        "intent": "sar",
        "all_intents": ["sar", "detect"],
        "supported": True,
        "model_used": "YOLOv8 SAR Vessel Detector",
        "answer": (
            "**SAR Maritime Analysis**\n\n"
            "Detected **3** maritime target(s) in SAR imagery:\n\n"
            "| # | Object | Confidence | Bounding Box |\n"
            "|---|--------|-----------|-------------|\n"
            "| 1 | Ship | 76.3% | [240, 243, 285, 297] |\n"
            "| 2 | Ship | 26.8% | [144, 1, 188, 26] |\n"
            "| 3 | Ship | 25.3% | [243, 244, 296, 307] |\n\n"
            "_Inference: ~50ms, GPU VRAM: 21 MB_\n\n"
            "**Context:** SAR (Synthetic Aperture Radar) imagery penetrates clouds "
            "and works day/night, making it ideal for maritime surveillance. "
            "This YOLOv8-based detector identifies vessels in SAR scenes using "
            "backscatter intensity patterns.\n\n"
            "*See annotated image for bounding box visualization.*"
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
    {
        "name": "❓ Urban VQA — Building Count",
        "image": os.path.join(ROOT, "test_images", "urban_optical.jpg"),
        "query": "Can you see any buildings or infrastructure in this area?",
        "intent": "vqa",
        "all_intents": ["vqa"],
        "supported": True,
        "model_used": "EarthDial 4B RGB (VLM)",
        "answer": (
            "### Building & Infrastructure Assessment\n\n"
            "**Yes**, this area contains significant built infrastructure:\n\n"
            "**Buildings:**\n"
            "- Multiple clustered residential structures visible throughout the scene\n"
            "- Building density suggests established urban/suburban settlement\n"
            "- Structures appear to be low-rise (1-3 stories) residential buildings\n\n"
            "**Infrastructure:**\n"
            "- Road network connecting building clusters\n"
            "- Linear features suggest planned street grid in some areas\n"
            "- Open spaces between buildings indicate courtyards or vacant lots\n\n"
            "**Assessment:** This is a developed area with moderate-to-high "
            "building density, consistent with peri-urban residential zones "
            "commonly found in Indian cities."
        ),
        "elapsed_vlm_s": 0.0,
        "elapsed_total_s": 0.0,
    },
]


def get_demo_list() -> list[str]:
    """Return list of demo names for the dropdown."""
    return [d["name"] for d in DEMOS]


def get_demo_by_name(name: str) -> dict | None:
    """Return demo dict by name, or None if not found."""
    for d in DEMOS:
        if d["name"] == name:
            return d
    return None
