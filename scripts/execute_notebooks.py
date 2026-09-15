"""Execute every notebook using this environment's interpreter, without user kernels."""
from pathlib import Path
import sys
import tempfile
import json
import os
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]


def execute_in_process():
    """Portable execution without subprocesses or sockets; notebooks contain Python only."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)
    records=[]
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        print('Executing in process',path.name,flush=True)
        InteractiveShell.clear_instance()
        shell=InteractiveShell.instance()
        notebook=nbformat.read(path,as_version=4)
        count=0
        for cell in notebook.cells:
            if cell.cell_type!='code':
                continue
            count+=1
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            result.raise_error()
            outputs=[]
            if captured.stdout:
                outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:
                outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for output in captured.outputs:
                outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
            cell.outputs=outputs
            cell.execution_count=count
        nbformat.write(notebook,path)
        records.append({'notebook':path.name,'status':'executed','code_cells':count,'runner':'IPython in-process'})
    (ROOT/'reports/notebook_execution.json').write_text(json.dumps(records,indent=2))


def main():
    # Temporary private kernelspec avoids registering or modifying global Jupyter settings.
    with tempfile.TemporaryDirectory() as temporary:
        kernels=Path(temporary)/'kernels'/'capstone'
        kernels.mkdir(parents=True)
        (kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Capstone','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[str(kernels.parent)])
        results=[]
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            print('Executing',path.name,flush=True)
            notebook=nbformat.read(path,as_version=4)
            client=NotebookClient(notebook,timeout=300,kernel_name='capstone',
                resources={'metadata':{'path':str(ROOT)}})
            client.create_kernel_manager()
            client.km.kernel_spec_manager=manager
            client.execute()
            nbformat.write(notebook,path)
            results.append({'notebook':path.name,'status':'executed','code_cells':sum(c.cell_type=='code' for c in notebook.cells)})
        (ROOT/'reports/notebook_execution.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    if '--in-process' in sys.argv:
        execute_in_process()
    else:
        main()
