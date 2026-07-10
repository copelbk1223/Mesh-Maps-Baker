import time
import traceback

import bpy

from .bake_pipeline import run_bake
from .maps import BakeError


class MESHBAKER_OT_bake(bpy.types.Operator):
    """Bake the enabled mesh maps (Esc to cancel)"""
    bl_idname = "mesh_baker.bake"
    bl_label = "Bake Maps"
    bl_options = {'REGISTER'}

    _timer = None
    _gen = None
    _results = None

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH'

    def invoke(self, context, event):
        s = context.scene.mesh_baker

        if not any((s.use_normal, s.use_ws_normal, s.use_ao, s.use_thickness,
                    s.use_curvature, s.use_position, s.use_matid, s.use_height)):
            self.report({'ERROR'}, "No maps enabled")
            return {'CANCELLED'}
        if s.save_to_disk and s.out_dir.startswith("//") and not bpy.data.filepath:
            self.report({'ERROR'},
                        "Output folder is relative (//) but the .blend is unsaved — "
                        "save the file or set an absolute output folder")
            return {'CANCELLED'}

        self._results = {}
        try:
            self._gen = run_bake(context, self._results)
        except BakeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set("Mesh Baker: starting…  (Esc to cancel)")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context)
            self.report({'WARNING'}, "Bake cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        wm = context.window_manager
        budget_end = time.time() + 0.15
        try:
            frac, label = 0.0, ""
            while time.time() < budget_end:
                frac, label = next(self._gen)
            wm.progress_update(int(frac * 100))
            context.workspace.status_text_set(
                f"Mesh Baker: {label}  —  {int(frac * 100)}%  (Esc to cancel)")
            return {'RUNNING_MODAL'}
        except StopIteration:
            self._finish(context)
            r = self._results
            for w in r.get("warnings", []):
                self.report({'WARNING'}, w)
            n = len(r.get("images", []))
            where = f" to {bpy.path.abspath(context.scene.mesh_baker.out_dir)}" \
                if context.scene.mesh_baker.save_to_disk else " (in blend file)"
            self.report({'INFO'},
                        f"Baked {n} map(s) in {r.get('time', 0):.1f}s "
                        f"[{r.get('engine', '?')} engine]{where}")
            return {'FINISHED'}
        except BakeError as e:
            self._finish(context)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception:
            self._finish(context)
            traceback.print_exc()
            self.report({'ERROR'}, "Bake failed — see the system console for details")
            return {'CANCELLED'}

    def _finish(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)
        self._gen = None


def register():
    bpy.utils.register_class(MESHBAKER_OT_bake)


def unregister():
    bpy.utils.unregister_class(MESHBAKER_OT_bake)
