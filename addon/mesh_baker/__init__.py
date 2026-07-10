bl_info = {
    "name": "Mesh Baker",
    "author": "Mesh Baker contributors",
    "version": (1, 0, 0),
    "blender": (4, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Mesh Baker",
    "description": "Substance/Marmoset-style raycast baker: Normal, AO, Curvature, "
                   "Thickness, Position, World Space Normal and ID maps. "
                   "High-to-low projection with cage support, or single object.",
    "category": "Baking",
}

# standard Blender reload pattern: on Reload Scripts, "bpy" is already in
# locals and the submodules must be re-imported freshly
if "bpy" in locals():
    import importlib
    importlib.reload(maps)
    importlib.reload(mesh_data)
    importlib.reload(engine_native)
    importlib.reload(engine_python)
    importlib.reload(bake_pipeline)
    importlib.reload(properties)
    importlib.reload(operators)
    importlib.reload(ui)
else:
    from . import maps, mesh_data, engine_native, engine_python
    from . import bake_pipeline, properties, operators, ui

import bpy  # noqa: F401,E402


def register():
    properties.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    properties.unregister()
