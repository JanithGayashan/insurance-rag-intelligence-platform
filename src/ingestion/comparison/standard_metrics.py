"""Reference-based document metrics, with explicit annotation scope and capabilities.

TEDS follows PubTabNet's tree-edit definition after documented HTML
canonicalization. Layout AP uses the official pycocotools COCO evaluator.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
import json
import math
from pathlib import Path
import re

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from src.ingestion.comparison.evaluation import normalize, tokens

VERSION = "parsing-reference-metrics-v1"
LABEL_THRESHOLD = 0.8
ORDER_THRESHOLD = 85
CATEGORIES = ("heading", "list_item", "table")
CAPABILITIES = {name: set(CATEGORIES) for name in ("docling", "unstructured", "llamaparse")}
CAPABILITIES.update({name: {"table"} for name in ("pymupdf", "pdfplumber")})
LABELS = {"section_header": "heading", "title": "heading", "heading": "heading",
          "listitem": "list_item", "list_item": "list_item", "list": "list_item", "table": "table"}


def canonical_text(text: str) -> str:
    return " ".join(tokens(text))


def validate_reference(reference: dict | None, page_count: int):
    if not reference:
        return
    def valid_page(value):
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= page_count
    def check_box(box):
        return (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in box)
                and box[0] < box[2] and box[1] < box[3])
    scoped = set()
    scope_by_page = {}
    for page in reference.get("annotation_pages", []):
        number = page.get("page_number")
        if not valid_page(number) or number in scoped or not page.get("complete_categories") or not set(page["complete_categories"]) <= set(CATEGORIES):
            raise ValueError("Annotation pages require unique valid pages and explicit exhaustive categories")
        scoped.add(number)
        scope_by_page[number]=set(page["complete_categories"])
    identifiers = set()
    for field_name in ("elements", "tables"):
        for item in reference.get(field_name, []):
            if (not valid_page(item.get("page_number")) or item["page_number"] not in scoped
                or not isinstance(item.get("id"), str) or not item["id"] or item["id"] in identifiers
                or item.get("kind", "table") not in {*CATEGORIES, "text"}
                or ("bbox" in item and not check_box(item["bbox"]))):
                raise ValueError("Reference elements/tables require unique IDs, annotated pages and valid normalized boxes")
            if field_name == "elements" and (not isinstance(item.get("text"), str) or not item["text"].strip()):
                raise ValueError("Reference element text must be nonempty")
            kind=item.get("kind","table")
            if kind in CATEGORIES and kind not in scope_by_page[item["page_number"]]:
                raise ValueError("Reference categories must be declared exhaustive for their page")
            if field_name == "tables" and not table_tree(item)[0].children[0].children:
                raise ValueError("Reference tables require nonempty rows/cells")
            identifiers.add(item["id"])
    numbers = set()
    for page in reference.get("reading_order", []):
        if not valid_page(page.get("page_number")) or page["page_number"] in numbers or len(page.get("anchors", [])) < 2:
            raise ValueError("Reading-order references require unique valid pages and at least two anchors")
        numbers.add(page["page_number"])
        seen = set()
        phrases = set()
        for anchor in page["anchors"]:
            if (not isinstance(anchor.get("id"), str) or not anchor["id"] or anchor["id"] in seen
                or not isinstance(anchor.get("text"), str) or len(tokens(anchor["text"])) < 2):
                raise ValueError("Reading-order anchors require unique IDs and at least two text tokens")
            phrase=canonical_text(anchor["text"])
            if phrase in phrases:
                raise ValueError("Reading-order anchor text must be unique within its page")
            phrases.add(phrase)
            seen.add(anchor["id"])


def load_evaluation_input(directory: Path, filename: str = "result.json") -> dict:
    result = json.loads((directory / filename).read_text(encoding="utf-8"))
    # Older Unstructured normalized outputs omitted the pixel coordinate space.
    # Recover it from the saved native response without reparsing the document.
    native = directory / "native.json"
    if result.get("parser_name") == "unstructured" and native.is_file():
        spaces = {}
        for item in json.loads(native.read_text(encoding="utf-8")):
            meta = item.get("metadata", {})
            coordinates = meta.get("coordinates", {})
            if coordinates.get("system") == "PixelSpace":
                spaces[meta.get("page_number")] = [coordinates.get("layout_width"), coordinates.get("layout_height")]
        result["_coordinate_sizes"] = spaces
    return result


def normalized_box(item: dict, result: dict, source_pages: list[dict]) -> list[float] | None:
    box, page_number = item.get("bbox"), item.get("page_number")
    if not box or len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return None
    profile = next((page for page in source_pages if page["page_number"] == page_number), {})
    parser = result.get("parser_name")
    if parser == "unstructured":
        size = result.get("_coordinate_sizes", {}).get(page_number, [])
    else:
        size = [profile.get("width"), profile.get("height")]
    if len(size) != 2 or not all(isinstance(v, (int, float)) and v > 0 for v in size):
        return None
    x0, y0, x1, y1 = box
    x0, x1 = sorted([x0 / size[0], x1 / size[0]])
    y0, y1 = sorted([y0 / size[1], y1 / size[1]])
    if parser == "docling":
        y0, y1 = 1 - y1, 1 - y0
    elif parser in {"pdfplumber", "pymupdf"} and profile.get("rotation", 0):
        return None  # Do not guess how a rotated parser's native box was represented.
    box = [max(0., min(1., value)) for value in [x0, y0, x1, y1]]
    return box if box[0] < box[2] and box[1] < box[3] else None


def predicted_elements(result: dict, source_pages: list[dict]) -> list[dict]:
    items = []
    previous_was_heading = False
    for index, item in enumerate(result.get("elements", [])):
        kind = LABELS.get(item["type"].lower())
        if kind not in {"heading", "list_item"}:
            previous_was_heading = False
            continue
        texts = [item["text"]]
        if item["type"].lower() == "list":
            texts = re.split(r"(?m)^\s*(?=(?:\d+[.)]|[a-zA-Z][.)]|[-*•])\s+)", item["text"])
            texts = [text.strip() for text in texts if text.strip()]
        for part, text in enumerate(texts):
            node = {"id": f"element-{index}-{part}", "page_number": item.get("page_number"),
                "kind": kind, "text": text, "bbox": normalized_box(item, result, source_pages),
                "confidence": item.get("metadata", {}).get("detection_class_prob", 1.0)}
            if (kind == "heading" and previous_was_heading and items and items[-1]["kind"] == kind
                and items[-1]["page_number"] == node["page_number"]):
                previous = items[-1]
                previous["text"] += "\n" + text
                if previous["bbox"] and node["bbox"]:
                    a, b = previous["bbox"], node["bbox"]
                    previous["bbox"] = [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                else:
                    previous["bbox"] = None
            else:
                items.append(node)
            previous_was_heading = kind == "heading"
    for index, item in enumerate(result.get("tables", [])):
        items.append({**item, "id": f"table-{index}", "kind": "table",
                      "bbox": normalized_box(item, result, source_pages), "confidence": 1.0})
    return items


def assignment(scores):
    if not scores or not scores[0]:
        return []
    rows, columns = linear_sum_assignment(np.asarray(scores), maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, columns)]


def label_similarity(expected: str, actual: str) -> float:
    def words(text):
        return Counter(tokens(re.sub(r"^\s*(?:\d+[.)]|[a-zA-Z][.)]|[-*•])\s+", "", text)))
    a, b = words(expected), words(actual)
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values())) if a and b else 0.


def label_metrics(predictions: list[dict], reference: dict, capabilities: set[str]) -> dict:
    result = {}
    for kind in ("heading", "list_item"):
        pages = {item["page_number"] for item in reference.get("annotation_pages", []) if kind in item["complete_categories"]}
        if kind not in capabilities or not pages:
            result[kind] = {"status": "NOT_SUPPORTED" if kind not in capabilities else "NOT_ANNOTATED",
                            "precision": None, "recall": None, "f1": None, "pages": sorted(pages)}
            continue
        expected = [item for item in reference.get("elements", []) if item["kind"] == kind and item["page_number"] in pages]
        observed = [item for item in predictions if item["kind"] == kind and item["page_number"] in pages]
        scores = [[label_similarity(a["text"], b["text"]) if a["page_number"] == b["page_number"] else 0
                   for b in observed] for a in expected]
        weights=[[1+value/(max(len(expected),len(observed))+1) if value>=LABEL_THRESHOLD else 0 for value in row] for row in scores]
        pairs = [(i,j) for i,j in assignment(weights) if scores[i][j] >= LABEL_THRESHOLD]
        tp, fp, fn = len(pairs), len(observed) - len(pairs), len(expected) - len(pairs)
        precision = tp / (tp + fp) if tp + fp else 0. if expected else None
        recall = tp / (tp + fn) if tp + fn else None
        result[kind] = {"status": "MEASURED" if expected or observed else "NO_POSITIVES", "pages": sorted(pages),
            "precision": precision, "recall": recall, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "tp": tp, "fp": fp, "fn": fn, "reference_count": len(expected), "prediction_count": len(observed),
            "matches": [{"reference_id": expected[i]["id"], "prediction_id": observed[j]["id"],
                         "similarity": scores[i][j]} for i,j in pairs],
            "missing": [a for i,a in enumerate(expected) if not any(i == p[0] for p in pairs)],
            "extra": [a for j,a in enumerate(observed) if not any(j == p[1] for p in pairs)]}
    return result


def reading_order_metrics(result: dict, reference: dict) -> dict:
    pages = {page["page_number"]: page.get("text", "") for page in result.get("pages", [])}
    rows = []
    for page in reference.get("reading_order", []):
        text = canonical_text(pages.get(page["page_number"], ""))
        matches = []
        for anchor in page["anchors"]:
            phrase = canonical_text(anchor["text"])
            exact = list(re.finditer(r"(?<!\w)"+re.escape(phrase)+r"(?!\w)", text))
            if exact:
                matches.extend({"id": anchor["id"], "position": found.start(), "similarity": 1.0} for found in exact)
            elif text:
                alignment = fuzz.partial_ratio_alignment(phrase, text, score_cutoff=ORDER_THRESHOLD)
                if alignment and label_similarity(phrase,text[alignment.dest_start:alignment.dest_end]) >= 0.5:
                    matches.append({"id": anchor["id"], "position": alignment.dest_start, "similarity": alignment.score/100})
        # Two fuzzy anchors cannot occupy the same position; keep the stronger match.
        positions = {}
        for match in sorted(matches, key=lambda item: item["similarity"], reverse=True):
            positions.setdefault(match["position"], match)
        matches = sorted(positions.values(), key=lambda item: item["position"])
        expected, actual = [a["id"] for a in page["anchors"]], [a["id"] for a in matches]
        distance = Levenshtein.distance(expected, actual)
        units = max(len(expected), len(actual))
        rows.append({"page_number": page["page_number"], "expected": expected, "predicted": actual,
            "matches": matches, "missing": [a for a in expected if a not in actual],
            "matched_anchors": len(set(actual)), "reference_anchors": len(expected),
            "edits": distance, "normalization_units": units,
            "normalized_edit_distance": distance / units if units else None})
    units = sum(row["normalization_units"] for row in rows)
    total = sum(row["reference_anchors"] for row in rows)
    return {"status": "MEASURED" if rows else "NOT_ANNOTATED", "pages": rows,
        "normalized_edit_distance": sum(row["edits"] for row in rows)/units if units else None,
        "anchor_coverage": sum(row["matched_anchors"] for row in rows)/total if total else None,
        "reference_anchors": total}


@dataclass
class TableNode:
    tag: str
    content: str = ""
    rowspan: int = 1
    colspan: int = 1
    children: list = field(default_factory=list)


def table_tree(table: dict) -> tuple[TableNode, str]:
    """Canonicalize row groups, th/td and inline formatting; retain cell spans."""
    from lxml import html
    body = TableNode("tbody")
    if table.get("html"):
        root = html.fromstring(table["html"], parser=html.HTMLParser(no_network=True, remove_comments=True))
        if root.tag != "table":
            roots = root.xpath(".//table")
            if not roots:
                raise ValueError("Table HTML contains no table")
            root = roots[0]
        for row in root.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):
            cells = []
            for cell in row.xpath("./td|./th"):
                rowspan, colspan = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
                if not 1 <= rowspan <= 1000 or not 1 <= colspan <= 1000:
                    raise ValueError("Table spans must be between 1 and 1000")
                cells.append(TableNode("td", normalize(" ".join(cell.itertext())), rowspan, colspan))
            if cells:
                body.children.append(TableNode("tr", children=cells))
        representation = "html_with_spans"
    else:
        for row in table.get("rows", []):
            if row:
                body.children.append(TableNode("tr", children=[TableNode("td", normalize(str(cell))) for cell in row]))
        representation = "rows_assuming_unit_spans"
    return TableNode("table", children=[body]), representation


def teds(expected: dict, actual: dict, *, structure_only: bool = False) -> float:
    from apted import APTED, Config
    class Costs(Config):
        def rename(self, a, b):
            if (a.tag, a.rowspan, a.colspan) != (b.tag, b.rowspan, b.colspan):
                return 1.
            length = max(len(a.content), len(b.content))
            return Levenshtein.distance(a.content,b.content)/length if a.tag == "td" and length and not structure_only else 0.
    a, _ = table_tree(expected)
    b, _ = table_tree(actual)
    if not a.children[0].children:
        raise ValueError("Reference table is empty")
    if not b.children[0].children:
        return 0.
    def descendants(node):
        return sum(1 + descendants(child) for child in node.children)
    return max(0., 1. - APTED(a, b, Costs()).compute_edit_distance()/max(descendants(a),descendants(b)))


def table_metrics(result: dict, reference: dict) -> dict:
    pages = {p["page_number"] for p in reference.get("annotation_pages", []) if "table" in p["complete_categories"]}
    if not pages:
        return {"status": "NOT_ANNOTATED", "teds": None, "structure_teds": None, "pages": []}
    rows = []
    for page in sorted(pages):
        expected = [t for t in reference.get("tables", []) if t["page_number"] == page]
        actual = [t for t in result.get("tables", []) if t.get("page_number") == page]
        scores, structures = [], []
        errors = {}
        for i,a in enumerate(expected):
            score_row, structure_row = [], []
            for j,b in enumerate(actual):
                try:
                    score_row.append(teds(a,b))
                    structure_row.append(teds(a,b,structure_only=True))
                except (ValueError, TypeError) as error:
                    score_row.append(0.)
                    structure_row.append(0.)
                    errors[j] = str(error)
            scores.append(score_row)
            structures.append(structure_row)
        pairs = assignment(scores)
        denominator = max(len(expected),len(actual))
        rows.append({"page_number": page, "reference_tables": len(expected), "predicted_tables": len(actual),
            "matched_tables":len(pairs),
            "normalization_units": denominator,
            "teds_sum": sum(scores[i][j] for i,j in pairs), "structure_teds_sum": sum(structures[i][j] for i,j in pairs),
            "missing_tables": max(0,len(expected)-len(actual)), "extra_tables": max(0,len(actual)-len(expected)),
            "matches": [{"reference_id": expected[i]["id"], "prediction_index_on_page": j,
                "teds": scores[i][j], "structure_teds": structures[i][j],
                "representation": "html_with_spans" if actual[j].get("html") else "rows_assuming_unit_spans"} for i,j in pairs],
            "invalid_tables": errors,
            "teds": sum(scores[i][j] for i,j in pairs)/denominator if denominator else None,
            "structure_teds": sum(structures[i][j] for i,j in pairs)/denominator if denominator else None})
    units = sum(row["normalization_units"] for row in rows)
    matched = sum(row["matched_tables"] for row in rows)
    return {"status": "MEASURED", "pages": rows, "reference_tables": sum(row["reference_tables"] for row in rows),
        "predicted_tables": sum(row["predicted_tables"] for row in rows),
        "matched_teds":sum(row["teds_sum"] for row in rows)/matched if matched else None,
        "teds": sum(row["teds_sum"] for row in rows)/units if units else None,
        "structure_teds": sum(row["structure_teds_sum"] for row in rows)/units if units else None}


def iou(a, b) -> float:
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def layout_metrics(predictions: list[dict], result: dict, reference: dict, capabilities: set[str]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    expected = [item for item in reference.get("elements", []) if item["kind"] in CATEGORIES]
    expected += [{**item,"kind":"table"} for item in reference.get("tables", [])]
    classes, details = [], {}
    for kind in CATEGORIES:
        pages = {p["page_number"] for p in reference.get("annotation_pages", []) if kind in p["complete_categories"]}
        truth = [p for p in expected if p["kind"] == kind and p["page_number"] in pages]
        predicted = [p for p in predictions if p["kind"] == kind and p["page_number"] in pages]
        status = "NOT_SUPPORTED" if kind not in capabilities or result.get("parser_name") == "llamaparse" and not any(p.get("bbox") for p in predicted) else (
            "NOT_ANNOTATED" if not truth else "MISSING_COORDINATES" if any(not p.get("bbox") for p in truth+predicted) else "MEASURED")
        details[kind] = {"status":status,"reference_count":len(truth),"prediction_count":len(predicted),
                         "ap50":None,"map_50_95":None,"mean_best_iou":None}
        if status == "MEASURED":
            scores = [[iou(a["bbox"],b["bbox"]) if a["page_number"]==b["page_number"] else 0. for b in predicted] for a in truth]
            details[kind]["mean_best_iou"] = sum(scores[i][j] for i,j in assignment(scores))/len(truth)
            classes.append(kind)
    if not classes:
        return {"status":"UNAVAILABLE","ap50":None,"map_50_95":None,"by_kind":details}
    category_ids = {kind:index+1 for index,kind in enumerate(classes)}
    pages = sorted({p["page_number"] for p in reference.get("annotation_pages", [])})
    scoped_pages={kind:{p["page_number"] for p in reference.get("annotation_pages",[]) if kind in p["complete_categories"]} for kind in classes}
    annotations, detections = [], []
    for collection, output in ((expected,annotations),(predictions,detections)):
        for item in collection:
            if item["kind"] not in classes or item["page_number"] not in scoped_pages[item["kind"]]:
                continue
            box=item.get("bbox")
            if not box:
                continue
            x0,y0,x1,y1 = [v*1000 for v in box]
            row={"image_id":item["page_number"],"category_id":category_ids[item["kind"]],"bbox":[x0,y0,x1-x0,y1-y0]}
            if output is annotations:
                row.update(id=len(output)+1,area=(x1-x0)*(y1-y0),iscrowd=0)
            else:
                confidence = item.get("confidence")
                row["score"] = float(confidence) if isinstance(confidence,(float,int)) and math.isfinite(confidence) else 1.
            output.append(row)
    with redirect_stdout(StringIO()):
        ground = COCO()
        ground.dataset={"info":{},"images":[{"id":p,"width":1000,"height":1000} for p in pages],
                        "categories":[{"id":i,"name":kind} for kind,i in category_ids.items()],"annotations":annotations}
        ground.createIndex()
        if detections:
            predicted = ground.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset={**ground.dataset,"annotations":[]}
            predicted.createIndex()
        evaluator = COCOeval(ground,predicted,"bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    for kind,index in category_ids.items():
        all_values = precision[:,:,index-1,0,-1]
        at50 = precision[0,:,index-1,0,-1]
        details[kind].update(map_50_95=float(all_values[all_values>=0].mean()),ap50=float(at50[at50>=0].mean()))
    return {"status":"MEASURED","ap50":float(evaluator.stats[1]),"map_50_95":float(evaluator.stats[0]),
        "by_kind":details,"evaluator":"pycocotools COCOeval bbox, IoU .50:.05:.95, 101 recall points, maxDets=100",
        "confidence_policy":"Use supplied detection_class_prob; otherwise score=1 with stable saved-output ordering."}


def standard_evaluation(result: dict, source_pages: list[dict], reference: dict | None) -> dict:
    if not reference:
        return {"version":VERSION,"status":"NOT_ANNOTATED"}
    if not result.get("success"):
        return {"version":VERSION,"status":"FAILED"}
    annotated={p["page_number"] for p in reference.get("annotation_pages",[])}
    predictions=[p for p in predicted_elements(result,source_pages) if p["page_number"] in annotated]
    capabilities=CAPABILITIES.get(result.get("parser_name"),{item["kind"] for item in predictions})
    return {"version":VERSION,"status":"EVALUATED","review_status":reference.get("review_status","NOT_PROVIDED"),
        "annotation_pages":[p["page_number"] for p in reference.get("annotation_pages",[])],
        "labels":label_metrics(predictions,reference,capabilities),"reading_order":reading_order_metrics(result,reference),
        "tables":table_metrics(result,reference),"layout":layout_metrics(predictions,result,reference,capabilities),
        "predicted_regions":predictions,
        "methodology":{"text":"WER/CER use complete reference transcriptions after Unicode NFKC, case and whitespace normalization. Recognized Markdown table syntax is removed for text scoring only. WER uses Unicode word tokens. CER/WER divide total edits by total reference units; normalized text edit distance uses the larger string length. These metrics cover annotated pages only.",
            "label_matching":"One-to-one maximum-weight same-page text-token F1 matching at >=0.8; list markers ignored. Consecutive heading lines form one block; list groups are split at markers.",
            "reading_order":"Normalized Levenshtein distance between reference anchor IDs and their occurrence order in page text. Exact matching, then fuzzy partial ratio >=85; missing/duplicate anchors affect distance. Inspect anchor coverage.",
            "tables":"TEDS and structure-only TEDS on canonical HTML: row groups unified, th becomes td, cell text NFKC/case/whitespace normalized, inline formatting removed; rowspan/colspan retained. Rows-only outputs assume unit spans. Best one-to-one table assignment per page; divide summed scores by max(reference,predicted) counts per page to penalize missing/extra tables.",
            "layout":"Official COCO bbox AP on exhaustively annotated heading/list/table classes and selected pages only. Native coordinates are converted to normalized top-left boxes. Missing coordinates remain unavailable, not zero accuracy.",
            "limitations":"One provisional document; agent annotations need human review. These results are not official OmniDocBench/DocLayNet scores or full-corpus accuracy."}}
