from copy import deepcopy

import pytest

from src.ingestion.comparison.standard_metrics import (
    label_metrics, layout_metrics, normalized_box, predicted_elements, reading_order_metrics,
    standard_evaluation, table_metrics, teds, validate_reference,
)


def reference():
    return {"review_status":"PENDING_HUMAN_REVIEW",
        "annotation_pages":[{"page_number":1,"complete_categories":["heading","list_item","table"]}],
        "elements":[{"id":"h1","page_number":1,"kind":"heading","text":"Motor coverage","bbox":[.1,.1,.8,.2]},
                    {"id":"l1","page_number":1,"kind":"list_item","text":"a. Report every claim promptly","bbox":[.1,.3,.8,.4]}],
        "tables":[{"id":"t1","page_number":1,"bbox":[.1,.5,.8,.9],"rows":[["Benefit","Limit"],["Glass","1500"]]}],
        "reading_order":[{"page_number":1,"anchors":[{"id":"h1","text":"Motor coverage"},
            {"id":"l1","text":"Report every claim promptly"}]}]}


def output():
    return {"parser_name":"pymupdf","success":True,"pages":[{"page_number":1,"text":"Motor coverage Report every claim promptly"}],
            "elements":[],"tables":[{"page_number":1,"bbox":[10,50,80,90],"rows":[["Benefit","Limit"],["Glass","1500"]]}]}


def test_label_precision_penalizes_duplicates_and_recall_penalizes_missing_labels():
    predicted=[{"id":"a","page_number":1,"kind":"heading","text":"Motor coverage"},
               {"id":"b","page_number":1,"kind":"heading","text":"Motor coverage"},
               {"id":"outside","page_number":2,"kind":"heading","text":"Not annotated"}]
    scores=label_metrics(predicted,reference(),{"heading","list_item"})
    assert scores["heading"]["precision"] == .5
    assert scores["heading"]["recall"] == 1
    assert scores["heading"]["f1"] == pytest.approx(2/3)
    assert scores["list_item"]["fn"] == 1
    assert scores["list_item"]["f1"] == 0
    assert label_metrics(predicted,reference(),set())["heading"]["f1"] is None


def test_list_groups_are_split_and_consecutive_heading_lines_are_combined():
    data=output()
    data["elements"]=[{"type":"heading","page_number":1,"text":"SECTION 3"},
        {"type":"heading","page_number":1,"text":"MEDICAL EXPENSES"},
        {"type":"list","page_number":1,"text":"1. Parent clause\n\n   a. Nested condition\n\n2. Next clause"}]
    actual=predicted_elements(data,[])
    assert actual[0]["text"] == "SECTION 3\nMEDICAL EXPENSES"
    assert len([p for p in actual if p["kind"]=="list_item"]) == 3


def test_order_changes_missing_anchors_and_duplicates_are_detected():
    data=output()
    assert reading_order_metrics(data,reference())["normalized_edit_distance"] == 0
    data["pages"][0]["text"]="Report every claim promptly Motor coverage"
    assert reading_order_metrics(data,reference())["normalized_edit_distance"] == 1
    data["pages"][0]["text"]="Motor coverage Motor coverage"
    score=reading_order_metrics(data,reference())
    assert score["anchor_coverage"] == .5
    assert score["normalized_edit_distance"] > 0
    data["pages"][0]["text"]=""
    assert reading_order_metrics(data,reference())["normalized_edit_distance"] == 1


def test_order_fuzzy_matching_does_not_invent_brand_text_from_a_website():
    ref={"reading_order":[{"page_number":1,"anchors":[{"id":"brand","text":"Sri Lanka Insurance"},
         {"id":"site","text":"www.srilankainsurance.com"}]}]}
    data={"pages":[{"page_number":1,"text":"www.srilankainsurance.com"}]}
    score=reading_order_metrics(data,ref)
    assert score["anchor_coverage"] == .5
    assert score["pages"][0]["missing"] == ["brand"]


def test_teds_uses_cell_content_structure_and_spans():
    truth={"html":"<table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>"}
    assert teds(truth,{"html":"<table><thead><tr><th>A</th><th>B</th></tr></thead></table>"}) == 1
    wrong={"rows":[["A","Wrong"]]}
    assert 0 < teds(truth,wrong) < 1
    assert teds(truth,wrong,structure_only=True) == 1
    span={"html":"<table><tr><td colspan='2'>A</td><td>B</td></tr></table>"}
    assert teds(truth,span,structure_only=True) < 1
    assert teds(truth,{"rows":[]}) == 0


def test_table_end_to_end_score_penalizes_extra_and_missing_tables():
    data=output()
    assert table_metrics(data,reference())["teds"] == 1
    data["tables"]*=2
    assert table_metrics(data,reference())["teds"] == .5
    data["tables"]=[]
    assert table_metrics(data,reference())["teds"] == 0
    data["tables"]=[{"page_number":2,"rows":[["outside"]]}]
    assert table_metrics(data,reference())["teds"] == 0


def test_official_coco_layout_is_perfect_for_matching_boxes_and_zero_for_wrong_boxes():
    data=output()
    profiles=[{"page_number":1,"width":100,"height":100}]
    predicted=predicted_elements(data,profiles)
    score=layout_metrics(predicted,data,reference(),{"table"})
    assert score["ap50"] == pytest.approx(1)
    assert score["map_50_95"] == pytest.approx(1)
    assert score["by_kind"]["table"]["mean_best_iou"] == 1
    predicted[0]["bbox"]=[.85,.05,.95,.1]
    assert layout_metrics(predicted,data,reference(),{"table"})["ap50"] == 0
    assert layout_metrics([],data,reference(),{"table"})["ap50"] == 0


def test_missing_coordinates_are_unavailable_and_class_scope_is_respected():
    data=output()
    predictions=predicted_elements(data,[])
    assert layout_metrics(predictions,data,reference(),{"table"})["ap50"] is None
    ref=reference()
    ref["annotation_pages"].append({"page_number":2,"complete_categories":["heading"]})
    data["tables"].append({"page_number":2,"bbox":[1,1,2,2],"rows":[["Not table-annotated"]]})
    predictions=predicted_elements(data,[{"page_number":p,"width":100,"height":100} for p in (1,2)])
    assert layout_metrics(predictions,data,ref,{"table"})["ap50"] == pytest.approx(1)


def test_native_and_pixel_box_coordinates_are_transformed_consistently():
    source=[{"page_number":1,"width":100,"height":200}]
    item={"page_number":1,"bbox":[10,160,80,180]}
    assert normalized_box(item,{"parser_name":"docling"},source) == pytest.approx([.1,.1,.8,.2])
    item["bbox"]=[20,40,160,80]
    assert normalized_box(item,{"parser_name":"unstructured","_coordinate_sizes":{1:[200,400]}},source) == [.1,.1,.8,.2]
    assert normalized_box(item,{"parser_name":"unstructured"},source) is None


@pytest.mark.parametrize("change",[
    lambda ref:ref["elements"][0].update(bbox=[0,0,2,1]),
    lambda ref:ref["elements"][0].update(bbox=[0,0,float('nan'),1]),
    lambda ref:ref["elements"].append(deepcopy(ref["elements"][0])),
    lambda ref:ref["reading_order"][0]["anchors"][1].update(id="h1"),
    lambda ref:ref["annotation_pages"][0].update(complete_categories=[]),
])
def test_invalid_or_incomplete_reference_annotations_are_rejected(change):
    ref=reference()
    change(ref)
    with pytest.raises(ValueError): validate_reference(ref,1)


def test_unannotated_and_failed_results_do_not_receive_zero_accuracy():
    assert standard_evaluation(output(),[],None)["status"] == "NOT_ANNOTATED"
    data=output()
    data["success"]=False
    assert standard_evaluation(data,[],reference())["status"] == "FAILED"
