import bpy
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, IntProperty,
    PointerProperty, StringProperty,
)


def _mesh_poll(self, obj):
    return obj.type == 'MESH'


class MeshBakerSettings(bpy.types.PropertyGroup):
    # ------------------------------------------------------------- mode
    mode: EnumProperty(
        name="Mode",
        items=[
            ('SINGLE', "Single Object", "Bake the active object onto itself"),
            ('H2L', "High to Low", "Project detail from high poly meshes onto the active (low poly) object"),
        ],
        default='H2L',
    )
    high_source: EnumProperty(
        name="High Poly Source",
        items=[
            ('SELECTED', "Other Selected", "All selected meshes except the active one"),
            ('COLLECTION', "Collection", "All meshes inside a collection"),
        ],
        default='SELECTED',
    )
    high_collection: PointerProperty(name="Collection", type=bpy.types.Collection)
    cage_object: PointerProperty(
        name="Cage", type=bpy.types.Object, poll=_mesh_poll,
        description="Optional cage mesh (same topology/UVs as the low poly). Rays are cast from the cage surface toward the low poly surface",
    )
    match_by_name: BoolProperty(
        name="Match By Name (_low/_high)", default=False,
        description="Only accept ray hits on high meshes whose base name matches the low poly (suffixes _low/_high, _lp/_hp)",
    )

    # ------------------------------------------------------- projection
    frontal_pct: FloatProperty(
        name="Max Frontal", default=2.0, min=0.0, max=100.0, subtype='PERCENTAGE',
        description="Ray search distance in front of the surface, as % of the bounding box diagonal",
    )
    rear_pct: FloatProperty(
        name="Max Rear", default=2.0, min=0.0, max=100.0, subtype='PERCENTAGE',
        description="Ray search distance behind the surface, as % of the bounding box diagonal",
    )

    # ----------------------------------------------------------- output
    resolution: EnumProperty(
        name="Resolution",
        items=[(str(s), f"{s} px", "") for s in (256, 512, 1024, 2048, 4096, 8192)],
        default='1024',
    )
    aa: EnumProperty(
        name="Antialiasing",
        items=[('1', "None", ""), ('2', "2x2 Subsampling", ""), ('4', "4x4 Subsampling", "")],
        default='1',
        description="Supersampling. Multiplies bake time by 4x (2x2) or 16x (4x4)",
    )
    dilation: IntProperty(
        name="Dilation", default=16, min=0, max=256,
        description="Edge padding in pixels flood-filled outside UV islands",
    )
    out_dir: StringProperty(
        name="Output Folder", subtype='DIR_PATH', default="//textures/",
    )
    base_name: StringProperty(
        name="Base Name", default="",
        description="File name prefix. Empty = active object name",
    )
    save_to_disk: BoolProperty(name="Save To Disk", default=True)
    engine: EnumProperty(
        name="Engine",
        items=[
            ('AUTO', "Auto", "Use the compiled native core if available, otherwise Python"),
            ('NATIVE', "Native (C++/Embree)", ""),
            ('PYTHON', "Python (slow fallback)", ""),
        ],
        default='AUTO',
    )
    half_res_rays: BoolProperty(
        name="Half Res AO/Thickness", default=True,
        description="Trace AO and thickness at half resolution then upscale (large speedup, like Substance's default)",
    )
    seed: IntProperty(name="Seed", default=0, min=0)

    # ------------------------------------------------------------- maps
    use_normal: BoolProperty(name="Normal (Tangent)", default=True)
    normal_directx: BoolProperty(
        name="DirectX (Y-)", default=False,
        description="Flip the green channel (DirectX convention). Off = OpenGL, what Blender expects",
    )
    use_ws_normal: BoolProperty(name="World Space Normal", default=False)

    use_ao: BoolProperty(name="Ambient Occlusion", default=True)
    ao_rays: IntProperty(name="Rays", default=64, min=4, max=1024)
    ao_distance_pct: FloatProperty(
        name="Distance", default=0.0, min=0.0, max=200.0, subtype='PERCENTAGE',
        description="Max occlusion distance as % of bounding box diagonal. 0 = unlimited",
    )
    ao_spread: FloatProperty(name="Spread Angle", default=180.0, min=1.0, max=180.0)
    ao_attenuation: EnumProperty(
        name="Attenuation",
        items=[('NONE', "None", ""), ('LINEAR', "Linear", ""), ('SMOOTH', "Smooth", "")],
        default='NONE',
    )
    ao_ignore_backface: BoolProperty(name="Ignore Backfaces", default=True)
    ao_self: EnumProperty(
        name="Self Occlusion",
        items=[
            ('ALWAYS', "Always", "Everything occludes everything"),
            ('SAME', "Only Same Mesh", "A mesh is only occluded by itself"),
            ('NEVER', "Never", "A mesh never occludes itself"),
        ],
        default='ALWAYS',
    )

    use_thickness: BoolProperty(name="Thickness", default=True)
    th_rays: IntProperty(name="Rays", default=64, min=4, max=1024)
    th_distance_pct: FloatProperty(
        name="Distance", default=25.0, min=0.1, max=200.0, subtype='PERCENTAGE',
        description="Normalization distance as % of bounding box diagonal",
    )
    th_spread: FloatProperty(name="Spread Angle", default=180.0, min=1.0, max=180.0)

    use_curvature: BoolProperty(name="Curvature", default=True)
    curv_intensity: FloatProperty(name="Intensity", default=1.0, min=0.01, max=20.0)
    curv_smooth: IntProperty(
        name="Smooth", default=0, min=0, max=8,
        description="Blur iterations applied to the normal data before deriving curvature",
    )
    curv_invert: BoolProperty(name="Invert", default=False)

    use_position: BoolProperty(name="Position", default=False)
    use_matid: BoolProperty(name="Material ID", default=True)
    id_source: EnumProperty(
        name="Color Source",
        items=[('MATERIAL', "Material Name", ""), ('MESH', "Mesh Name", "")],
        default='MATERIAL',
    )

    fmt_data: EnumProperty(
        name="Format",
        items=[('PNG', "PNG (8-bit)", ""), ('OPEN_EXR', "EXR (32-bit float)", "")],
        default='PNG',
        description="Position always saves as EXR regardless of this setting",
    )


def register():
    bpy.utils.register_class(MeshBakerSettings)
    bpy.types.Scene.mesh_baker = PointerProperty(type=MeshBakerSettings)


def unregister():
    del bpy.types.Scene.mesh_baker
    bpy.utils.unregister_class(MeshBakerSettings)
