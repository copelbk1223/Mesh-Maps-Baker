import bpy

from . import engine_native


class MESHBAKER_PT_panel(bpy.types.Panel):
    bl_label = "Mesh Baker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mesh Baker"

    def draw(self, context):
        lay = self.layout
        s = context.scene.mesh_baker

        ok, msg = engine_native.status()
        row = lay.row()
        row.label(text=msg, icon='CHECKMARK' if ok else 'ERROR')

        box = lay.box()
        box.label(text="Setup", icon='OBJECT_DATA')
        box.prop(s, "mode", expand=True)
        ob = context.active_object
        box.label(text=f"Low poly: {ob.name if ob else '—'}", icon='MESH_DATA')
        if s.mode == 'H2L':
            box.prop(s, "high_source", text="High Poly")
            if s.high_source == 'COLLECTION':
                box.prop(s, "high_collection", text="")
            box.prop(s, "cage_object")
            box.prop(s, "match_by_name")
            box.prop(s, "proj_ignore_backface")
            col = box.column(align=True)
            col.prop(s, "frontal_pct")
            col.prop(s, "rear_pct")

        box = lay.box()
        box.label(text="Maps", icon='TEXTURE')

        col = box.column(align=True)
        col.prop(s, "use_normal")
        if s.use_normal:
            col.prop(s, "normal_directx", toggle=True)
        col.separator()
        col.prop(s, "use_ao")
        if s.use_ao:
            sub = col.column(align=True)
            sub.prop(s, "ao_rays")
            sub.prop(s, "ao_distance_pct")
            sub.prop(s, "ao_spread")
            sub.prop(s, "ao_attenuation")
            sub.prop(s, "ao_ignore_backface")
            if s.mode == 'H2L':
                sub.prop(s, "ao_self")
        col.separator()
        col.prop(s, "use_curvature")
        if s.use_curvature:
            sub = col.column(align=True)
            sub.prop(s, "curv_source")
            sub.prop(s, "curv_intensity")
            sub.prop(s, "curv_auto_tonemap")
            sub.prop(s, "curv_smooth")
            sub.prop(s, "curv_invert")
        col.separator()
        col.prop(s, "use_thickness")
        if s.use_thickness:
            sub = col.column(align=True)
            sub.prop(s, "th_rays")
            sub.prop(s, "th_distance_pct")
            sub.prop(s, "th_spread")
        col.separator()
        col.prop(s, "use_ws_normal")
        col.prop(s, "use_position")
        if s.use_position:
            sub = col.column(align=True)
            sub.prop(s, "pos_mode")
            sub.prop(s, "pos_normalization")
        if s.mode == 'H2L':
            col.prop(s, "use_height")
            if s.use_height:
                col.prop(s, "height_scale_pct")
        col.prop(s, "use_matid")
        if s.use_matid:
            col.prop(s, "id_source")

        box = lay.box()
        box.label(text="Output", icon='OUTPUT')
        col = box.column(align=True)
        col.prop(s, "resolution")
        col.prop(s, "aa")
        col.prop(s, "dilation")
        col.prop(s, "half_res_rays")
        col.prop(s, "fmt_data")
        col.prop(s, "engine")
        col.prop(s, "seed")
        col.separator()
        col.prop(s, "save_to_disk")
        if s.save_to_disk:
            col.prop(s, "out_dir", text="")
        col.prop(s, "base_name")

        lay.separator()
        row = lay.row()
        row.scale_y = 1.6
        row.operator("mesh_baker.bake", icon='RENDER_STILL')


def register():
    bpy.utils.register_class(MESHBAKER_PT_panel)


def unregister():
    bpy.utils.unregister_class(MESHBAKER_PT_panel)
