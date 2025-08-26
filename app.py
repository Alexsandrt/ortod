# app.py — web-просмотрщик (trame + PyVista off-screen)
import os, re, numpy as np, pyvista as pv
from trame.app import get_server
from trame.widgets import html, vuetify as v
from trame.ui.vuetify import SinglePageWithDrawerLayout

# Установка OFF-SCREEN рендеринга для избежания предупреждений VTK
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
pv.OFF_SCREEN = True

DATA_DIR = os.environ.get("DATA_DIR", "/srv/jaw-viewer-data")


def extract_id(name):
    match = re.match(r"^0*(\d+)", name)
    return int(match.group(1)) if match else None


def detect_role(name):
    n = name.lower()
    if any(k in n for k in ["upper", "верх", "verh", "top", "_u"]):
        return "upper"
    elif any(k in n for k in ["lower", "низ", "niz", "bottom", "_l"]):
        return "lower"
    return None


def get_center_of_mass(mesh):
    return np.array(mesh.center)


def bbox_size(mesh):
    bounds = mesh.bounds
    return np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    ])


def normalize_scale(upper, lower):
    upper_size, lower_size = bbox_size(upper), bbox_size(lower)
    scale_factor = upper_size / np.where(lower_size == 0, 1e-9, lower_size)
    return lower.scale(scale_factor)


def prep_pair(upper, lower):
    lower = normalize_scale(upper, lower)
    translation_vector = get_center_of_mass(upper) - get_center_of_mass(lower)
    translation_vector[1] -= 8
    lower = lower.translate(translation_vector)
    upper = upper.translate([-1, +1, +3])
    return upper, lower


def mask_curv(mesh, quantile=0.65):
    curvature = mesh.curvature('mean')
    finite_values = curvature[np.isfinite(curvature)]
    if finite_values.size == 0:
        return None
    threshold = np.quantile(finite_values, quantile)
    return np.abs(mesh.curvature('mean')) >= threshold


def mask_height(mesh, quantile=0.60):
    z_coords = mesh.points[:, 2]
    finite_z = z_coords[np.isfinite(z_coords)]
    if finite_z.size == 0:
        return None
    threshold = np.quantile(finite_z, quantile)
    return mesh.points[:, 2] >= threshold


def colorize_teeth_gums(mesh, tooth_color=(255, 255, 255), gum_color=(242, 153, 153)):
    mask = mask_curv(mesh) or mask_height(mesh)
    if mask is None or mask.sum() in (0, mask.size):
        points_ordered_by_z = np.argsort(mesh.points[:, 2])
        mask = np.zeros(len(points_ordered_by_z), dtype=bool)
        if len(points_ordered_by_z) > 0:
            mask[points_ordered_by_z[-max(1, int(0.4 * len(points_ordered_by_z))):]] = True

    colors = np.empty((mesh.n_points, 3), dtype=np.uint8)
    colors[:] = gum_color
    colors[mask] = tooth_color
    mesh.point_data["RGB"] = colors
    return mesh


def collect_pairs(folder):
    stl_files = [
        filename for filename in os.listdir(folder)
        if filename.lower().endswith('.stl')
    ]
    model_buckets = {}
    for filename in stl_files:
        patient_id = extract_id(filename)
        role = detect_role(filename)
        if patient_id is None or role is None:
            continue
        model_buckets.setdefault(
            patient_id,
            {'upper': [], 'lower': []}
        )[role].append(filename)

    pairs = []
    for patient_id in sorted(model_buckets.keys()):
        group = model_buckets[patient_id]
        if not group['upper'] or not group['lower']:
            continue

        upper_mesh = pv.read(os.path.join(folder, group['upper'][0]))
        lower_mesh = pv.read(os.path.join(folder, group['lower'][0]))
        prepared_upper, prepared_lower = prep_pair(upper_mesh, lower_mesh)
        colored_upper = colorize_teeth_gums(prepared_upper)
        colored_lower = colorize_teeth_gums(prepared_lower)
        pairs.append((colored_upper, colored_lower, patient_id))

    return pairs


server = get_server(client_type='vue2')
state, ctrl = server.state, server.controller

pairs = collect_pairs(DATA_DIR)
state.pairs_len = len(pairs)
state.idx = 0
state.pid = pairs[0][2] if pairs else -1

plotter = pv.Plotter(off_screen=True)
plotter.set_background('white')


def render_idx(index):
    global plotter
    plotter.clear()
    if not pairs:
        return None
    index = max(0, min(len(pairs) - 1, index))
    upper_mesh, lower_mesh, patient_id = pairs[index]
    plotter.add_mesh(upper_mesh, scalars="RGB", rgb=True)
    plotter.add_mesh(lower_mesh, scalars="RGB", rgb=True)
    plotter.camera_position = "xy"
    image = plotter.screenshot(return_img=True)
    state.idx, state.pid = index, patient_id
    return image


@ctrl.trigger('next')
def next_pair():
    rendered_image = render_idx(state.idx + 1)
    if rendered_image is not None:
        ctrl.set_image('view', rendered_image)


@ctrl.trigger('prev')
def prev_pair():
    rendered_image = render_idx(state.idx - 1)
    if rendered_image is not None:
        ctrl.set_image('view', rendered_image)


@ctrl.trigger('jump')
def jump_to(index):
    try:
        index = int(index)
    except ValueError:
        return
    rendered_image = render_idx(index)
    if rendered_image is not None:
        ctrl.set_image('view', rendered_image)


initial_image = render_idx(0)

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Jaw Viewer")
    with layout.toolbar:
        v.VBtn("⟵ Previous", click=ctrl.prev, class_="mx-2", outlined=True)
        v.VBtn("Next ⟶", click=ctrl.next, class_="mx-2", outlined=True)
        v.Spacer()
        html.Div("Pair: {{ pid }}  [{{ idx+1 }}/{{ pairs_len }}]", class_="mx-2")
        v.TextField(v_model=("jump_to_idx", 0), label="Go to index", type="number", class_="mx-2",
                    style="max-width:120px;")
        v.VBtn("Go", click=lambda event: ctrl.jump(event, "jump_to_idx"), class_="mx-2", outlined=True)
    with layout.content:
        html.Img(src=("view", initial_image),
                 style="width:100%;max-width:1200px;border:1px solid #ddd;border-radius:8px")

if __name__ == "__main__":
    server.start(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))