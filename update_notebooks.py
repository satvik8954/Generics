import json
import os

def update_notebook_from_file(nb_path, py_path, markdown_match, is_exact=False):
    if not os.path.exists(nb_path) or not os.path.exists(py_path):
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    with open(py_path, "r", encoding="utf-8") as f:
        py_content = f.read()

    lines = [line + "\n" for line in py_content.split("\n")]
    if lines: lines[-1] = lines[-1].rstrip("\n")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            matches = False
            for line in cell["source"]:
                if is_exact:
                    if markdown_match == line.strip(): matches = True
                else:
                    if markdown_match in line: matches = True
            
            if matches:
                # The next code cell should be updated
                code_cell = nb["cells"][i+1]
                if code_cell["cell_type"] == "code":
                    code_cell["source"] = lines
                    break
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

# Update ExciPick_Pipeline.ipynb
update_notebook_from_file("ExciPick_Pipeline.ipynb", "config.py", "## config.py")
# The FULL_MODEL cell in pipeline has ExciPickHGNN, we don't need to replace it here directly unless we also updated FULL_MODEL.py. Wait, we updated excipient_scorer.py!
# But ExciPick_Pipeline.ipynb might only have FULL_MODEL.py or separate cells for model. Let's not blindly replace training unless we know the cell header.
# Actually we know it has ## training.py
update_notebook_from_file("ExciPick_Pipeline.ipynb", "training.py", "## training.py")
