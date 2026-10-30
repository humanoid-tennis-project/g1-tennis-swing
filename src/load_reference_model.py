"""Load local reference meshes through MuJoCo VFS, avoiding Windows path limits.

Mesh bytes and XML model parameters are unchanged. Only virtual filenames differ.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco


def load_reference_model(reference):
    reference=Path(reference).resolve()
    tree=ET.parse(reference/'ik_model_local.xml')
    root=tree.getroot();compiler=root.find('compiler')
    meshdir=Path(compiler.get('meshdir',''))
    assets={}
    for index,mesh in enumerate(root.findall('.//asset/mesh')):
        filename=mesh.get('file')
        if filename is None:continue
        source=(reference/meshdir/filename).resolve()
        virtual=f'mesh_{index}{source.suffix}'
        assets[virtual]=source.read_bytes()
        mesh.set('file',virtual)
    compiler.attrib.pop('meshdir',None)
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets=assets)
