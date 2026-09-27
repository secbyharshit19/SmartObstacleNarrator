from pathlib import Path
import nbformat

def test_notebook_schema_and_python():
    for p in (Path(__file__).parents[1]/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        for i,cell in enumerate(nb.cells):
            if cell.cell_type=='code':compile(cell.source,f'{p.name}:cell{i}','exec')
