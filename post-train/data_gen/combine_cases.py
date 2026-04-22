import json
import glob
import os
import shutil

# ---- CONFIG ----
input_files = sorted(glob.glob("training_cases_*.json"))
output_dir = "rft_data"
output_json = os.path.join(output_dir, "combined_training_cases.json")

# Ensure base dirs
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "transformation"), exist_ok=True)

# Priority order: higher value = higher priority
priority_order = {"unpivot": 4, "pivot": 3, "transpose": 2, "join": 1}

def get_transformation_type(case: dict) -> str:
    """Return the transformation key ('unpivot', 'pivot', etc.) or None."""
    transf = case.get("transformation", {})
    if not transf:
        return "join"
    return next(iter(transf.keys()), None)

def get_priority(case: dict) -> int:
    """Return numeric priority for a case."""
    t = get_transformation_type(case)
    return priority_order.get(t, 0)

combined = {}
sources = {}  # track which folder provided the winning case

for fpath in input_files:
    with open(fpath, "r") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{fpath} is not a JSON object")

        for case_id, case in data.items():
            if case_id not in combined or get_priority(case) > get_priority(combined[case_id]):
                combined[case_id] = case
                # remember which folder this case came from
                sources[case_id] = get_transformation_type(case)

# ---- Copy files for the chosen variant only ----
for case_id, case in combined.items():
    transf_type = get_transformation_type(case)

    # Copy case folder from the chosen source (if exists)
    # if transf_type:
    #     src_dir = os.path.join(f"cases_{transf_type}", case_id)
    #     if os.path.exists(src_dir):
    #         dst_dir = os.path.join(output_dir, case_id)
    #         if os.path.exists(dst_dir):
    #             shutil.rmtree(dst_dir)
    #         shutil.copytree(src_dir, dst_dir)

    #     # Copy transformation subdir if exists
    #     transf_src = os.path.join(f"cases_{transf_type}", "transformation", case_id, transf_type)
    #     if os.path.exists(transf_src):
    #         transf_dst = os.path.join(output_dir, "transformation", case_id, transf_type)
    #         if os.path.exists(transf_dst):
    #             shutil.rmtree(transf_dst)
    #         os.makedirs(os.path.dirname(transf_dst), exist_ok=True)
    #         shutil.copytree(transf_src, transf_dst)

    # Copy ground truth file if available
    gt_src = os.path.join(f"cases_{transf_type}/gt", f"{case_id}.csv")
    gt_dst = os.path.join(output_dir, "gt", f"{case_id}.csv")
    shutil.copy2(gt_src, gt_dst)

# ---- Write merged JSON ----
with open(output_json, "w") as out:
    json.dump(combined, out, indent=2)

print(f"✅ Merged {len(input_files)} files")
print(f"📦 Final number of unique cases: {len(combined)}")
print(f"📂 Output saved in {output_dir}")