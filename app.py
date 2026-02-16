import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from io import BytesIO, StringIO
from scipy.interpolate import make_interp_spline, Akima1DInterpolator, PchipInterpolator
from matplotlib.ticker import AutoMinorLocator

# --- Global Plotting Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # <--- השינוי שביקשת
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4
})

# --- Standard Colors ---
STANDARD_COLORS = {
    "Blue": "#0072BD", "Orange": "#D95319", "Yellow": "#EDB120",
    "Purple": "#7E2F8E", "Green": "#77AC30", "Cyan": "#4DBEEE",
    "Maroon": "#A2142F", "Black": "#000000", "Red": "#FF0000",
    "Dark Blue": "#0000FF", "Dark Green": "#008000"
}


# --- Helper Functions ---
def calculate_trendline(x, y, type_name):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2: return None, None, ""

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    try:
        if type_name == "Linear":
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
            eq = f"y = {z[0]:.2f}x + {z[1]:.2f}"
        elif type_name == "Exponential":
            pos_mask = y > 0
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(x_f, np.log(y_f), 1)
            A = np.exp(z[1])
            B = z[0]
            y_pred = A * np.exp(B * x_f)
            eq = f"y = {A:.2f}e^({B:.2f}x)"
            x, y = x_f, y_f
        elif type_name == "Logarithmic":
            pos_mask = x > 0
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(np.log(x_f), y_f, 1)
            A = z[0]
            B = z[1]
            y_pred = A * np.log(x_f) + B
            eq = f"y = {A:.2f}ln(x) + {B:.2f}"
            x, y = x_f, y_f
        elif type_name == "Power":
            pos_mask = (x > 0) & (y > 0)
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(np.log(x_f), np.log(y_f), 1)
            A = np.exp(z[1])
            B = z[0]
            y_pred = A * (x_f ** B)
            eq = f"y = {A:.2f}x^({B:.2f})"
            x, y = x_f, y_f
        elif type_name == "Polynomial (2nd Order)":
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            y_pred = p(x)
            eq = f"y = {z[0]:.2f}x² + {z[1]:.2f}x + {z[2]:.2f}"
        else:
            return None, None, ""

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return x, y_pred, f"{eq}\n($R^2={r2:.3f}$)"
    except:
        return None, None, "Fit Error"


def get_smooth_curve(x_in, y_in, algo):
    sort_idx = np.argsort(x_in)
    x_s = x_in[sort_idx]
    y_s = y_in[sort_idx]
    unique_x, indices = np.unique(x_s, return_index=True)
    unique_y = y_s[indices]
    x_new = np.linspace(unique_x.min(), unique_x.max(), 300)
    try:
        if len(unique_x) > 3:
            if algo == "Akima (Tight)":
                spl = Akima1DInterpolator(unique_x, unique_y)
                return x_new, spl(x_new)
            elif algo == "PCHIP (Monotonic)":
                spl = PchipInterpolator(unique_x, unique_y)
                return x_new, spl(x_new)
            else:
                spl = make_interp_spline(unique_x, unique_y, k=3)
                return x_new, spl(x_new)
        return unique_x, unique_y
    except:
        return unique_x, unique_y


# Helpers
def list_to_str(l): return ",".join(map(str, l)) if l else ""


def str_to_list(s): return [float(x) for x in s.split(',')] if ',' in s else None


def def_idx(val, lst): return lst.index(val) if val in lst else 0


# Sync color callback (Fixes color picker issue)
def sync_color(uid):
    preset_name = st.session_state[f"pst_{uid}"]
    st.session_state[f"clr_{uid}"] = STANDARD_COLORS[preset_name]


# --- Main Plotting Function ---
def create_advanced_plot(series_list, plot_settings):
    # Sort by User Order
    series_list.sort(key=lambda x: x['order'])
    fig, ax = plt.subplots(figsize=(12, 8))

    ax2 = None
    has_secondary = any(s['axis'] == 'Secondary' for s in series_list)
    if has_secondary:
        ax2 = ax.twinx()

    lines = []
    labels = []

    for series in series_list:
        # Extract and Clean Data
        x = series['x_data']
        y = series['y_data']
        y_err = series['err_data']

        # Cleaning Logic
        df_temp = pd.DataFrame({'x': x, 'y': y})
        df_temp['x'] = pd.to_numeric(df_temp['x'], errors='coerce')
        df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
        df_temp = df_temp.dropna()
        df_temp = df_temp.groupby('x', as_index=False).mean()
        df_temp = df_temp.sort_values('x')
        x = df_temp['x'].values
        y = df_temp['y'].values

        current_ax = ax2 if (series['axis'] == 'Secondary' and ax2) else ax

        color = series['color']
        label = series['label']
        ls = series['linestyle']
        lw = series['linewidth']
        mk = series['marker']
        ms = series['marker_size']
        mfc = 'white' if series['marker_fill'] == 'Hollow' else color
        mec = color

        if y_err is not None:
            if len(y_err) == len(x):  # Basic length check
                if series['error_style'] == "Sleeve":
                    y_lower = y - y_err
                    y_upper = y + y_err
                    if series['smoothing']:
                        px_l, py_l = get_smooth_curve(x, y_lower, series['smooth_algo'])
                        px_u, py_u = get_smooth_curve(x, y_upper, series['smooth_algo'])
                        current_ax.fill_between(px_l, py_l, py_u, color=color, alpha=series['sleeve_alpha'],
                                                linewidth=0)
                    else:
                        current_ax.fill_between(x, y_lower, y_upper, color=color, alpha=series['sleeve_alpha'],
                                                linewidth=0)
                else:
                    current_ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor=color, elinewidth=1.5,
                                        capsize=series['capsize'])

        if series['smoothing']:
            px, py = get_smooth_curve(x, y, series['smooth_algo'])
            if ls != "None":
                current_ax.plot(px, py, color=color, linestyle=ls, linewidth=lw)
            if mk != "None":
                current_ax.plot(x, y, linestyle='None', marker=mk, markersize=ms, markeredgecolor=mec,
                                markerfacecolor=mfc, markeredgewidth=2)
        else:
            final_ls = ls if ls != "None" else "none"
            final_mk = mk if mk != "None" else "none"
            current_ax.plot(x, y, color=color, linestyle=final_ls, linewidth=lw, marker=final_mk, markersize=ms,
                            markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2)

        if series['trendline'] != "None":
            tx, ty, t_label = calculate_trendline(x, y, series['trendline'])
            if tx is not None:
                final_t_label = f"{t_label}" if series['show_r2'] else None
                current_ax.plot(tx, ty, color=color, linestyle=':', linewidth=1.5, alpha=0.8, label=final_t_label)

        l, = current_ax.plot([], [], color=color, linestyle=ls if ls != "None" else "None", linewidth=lw,
                             marker=mk if mk != "None" else "None", markersize=ms, markeredgecolor=mec,
                             markerfacecolor=mfc, markeredgewidth=2, label=label)
        lines.append(l)
        labels.append(label)

    # --- Global Axis Settings ---
    ax.set_xlabel(plot_settings['x_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
    ax.set_ylabel(plot_settings['y_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")

    # X Limits
    if plot_settings['x_lim']:
        if len(plot_settings['x_lim']) == 3:
            start, step, end = plot_settings['x_lim']
            ax.set_xlim(start, end)
            ticks = np.arange(start, end + step / 100, step)
            ax.set_xticks(ticks)
        elif len(plot_settings['x_lim']) == 2:
            ax.set_xlim(plot_settings['x_lim'])

    # Y Limits Left
    if plot_settings['y_lim']:
        if len(plot_settings['y_lim']) == 3:
            start, step, end = plot_settings['y_lim']
            ax.set_ylim(start, end)
            ticks = np.arange(start, end + step / 100, step)
            ax.set_yticks(ticks)
        elif len(plot_settings['y_lim']) == 2:
            ax.set_ylim(plot_settings['y_lim'])

    # Y Limits Right
    if ax2:
        if plot_settings['y_label_right']:
            ax2.set_ylabel(plot_settings['y_label_right'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
        ax2.tick_params(axis='y', which='major', labelsize=plot_settings['fs_ticks'])
        if plot_settings['y_lim_right']:
            if len(plot_settings['y_lim_right']) == 3:
                start, step, end = plot_settings['y_lim_right']
                ax2.set_ylim(start, end)
                ticks = np.arange(start, end + step / 100, step)
                ax2.set_yticks(ticks)
            elif len(plot_settings['y_lim_right']) == 2:
                ax2.set_ylim(plot_settings['y_lim_right'])
        if plot_settings['minor_ticks_y'] > 0:
            ax2.yaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_y'] + 1))

    if plot_settings['title']:
        ax.set_title(plot_settings['title'], fontsize=plot_settings['fs_title'], fontname="Times New Roman", pad=15)

    ax.tick_params(axis='both', which='major', labelsize=plot_settings['fs_ticks'])
    if plot_settings['show_grid']: ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    if plot_settings['minor_ticks_x'] > 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_x'] + 1))
    if plot_settings['minor_ticks_y'] > 0:
        ax.yaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_y'] + 1))
    if plot_settings['minor_ticks_x'] > 0 or plot_settings['minor_ticks_y'] > 0:
        ax.tick_params(which='minor', direction='in', top=True, right=True)

    if plot_settings['legend_loc'] != "None":
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        extra_h = [];
        extra_l = []
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            if l not in labels and l is not None: extra_h.append(h); extra_l.append(l)
        final_handles = lines + extra_h
        final_labels = labels + extra_l
        ax.legend(final_handles, final_labels, loc=plot_settings['legend_loc'], fontsize=plot_settings['fs_legend'],
                  frameon=plot_settings['legend_frame'], fancybox=False, edgecolor='black',
                  ncol=plot_settings['legend_cols'])

    return fig


# --- PROJECT SAVING LOGIC (Fixed to save Styles) ---
def get_project_json():
    project = {
        "global_settings": st.session_state.get('loaded_config', {}),
        "series": []
    }
    # Iterate over series and capture current widget states from session_state
    for s in st.session_state.series_data:
        uid = s['internal_id']  # Use the Unique ID

        # Capture current styles from widgets or fallback to defaults
        style_data = {
            "label": st.session_state.get(f"lbl_{uid}", s.get("y_col_name")),
            "order": st.session_state.get(f"ord_{uid}", 0),
            "err_col": st.session_state.get(f"ecn_{uid}", "None"),
            "err_style": st.session_state.get(f"est_{uid}", "Bar"),
            "err_cap": st.session_state.get(f"ecs_{uid}", 4),
            "err_alpha": st.session_state.get(f"esa_{uid}", 0.2),
            "axis": st.session_state.get(f"ax_{uid}", "Left"),
            "preset_color": st.session_state.get(f"pst_{uid}", "Blue"),
            "hex_color": st.session_state.get(f"clr_{uid}", "#0072BD"),
            "linestyle": st.session_state.get(f"ls_{uid}", "-"),
            "linewidth": st.session_state.get(f"lw_{uid}", 2.0),
            "marker": st.session_state.get(f"mk_{uid}", "None"),
            "fill": st.session_state.get(f"mf_{uid}", "Solid"),
            "size": st.session_state.get(f"ms_{uid}", 8),
            "smooth": st.session_state.get(f"sm_{uid}", False),
            "smooth_algo": st.session_state.get(f"sa_{uid}", "Spline (Rounded)"),
            "trendline": st.session_state.get(f"tr_{uid}", "None")
        }

        df_main_str = s['df'].to_csv(index=False)
        df_err_str = s['df_err'].to_csv(index=False) if s['df_err'] is not None else None

        item = {
            "data_name": s['data_name'], "y_col_name": s['y_col_name'], "x_col_name": s['x_col_name'],
            "csv_data": df_main_str, "csv_err": df_err_str,
            "style": style_data  # Save the style dict
        }
        project['series'].append(item)
    return json.dumps(project)


# --- UI START ---
st.set_page_config(page_title="Pro Plotter", layout="wide")
st.title("📈 Pro Plotter")

# Initialize Session State
if 'series_data' not in st.session_state: st.session_state.series_data = []
if 'loaded_config' not in st.session_state: st.session_state.loaded_config = {}
if 'next_id' not in st.session_state: st.session_state.next_id = 0  # Counter for unique IDs

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Project Management")
    uploaded_proj = st.file_uploader("Load Project (.json)", type=["json"], key="loader")
    if uploaded_proj:
        try:
            proj_data = json.load(uploaded_proj)
            st.session_state.loaded_config = proj_data.get('global_settings', {})
            new_series = []

            # Reset ID counter based on loaded project to avoid clashes
            max_id = 0

            for i, s in enumerate(proj_data.get('series', [])):
                df_main = pd.read_csv(StringIO(s['csv_data']))
                df_err = pd.read_csv(StringIO(s['csv_err'])) if s['csv_err'] else None

                # Assign a new Unique ID
                current_uid = i
                if current_uid >= max_id: max_id = current_uid

                # Inject Styles back into Session State for Widgets
                style = s.get('style', {})
                st.session_state[f"lbl_{current_uid}"] = style.get('label', s['y_col_name'])
                st.session_state[f"ord_{current_uid}"] = style.get('order', i)
                st.session_state[f"ecn_{current_uid}"] = style.get('err_col', "None")
                st.session_state[f"est_{current_uid}"] = style.get('err_style', "Bar")
                st.session_state[f"ecs_{current_uid}"] = style.get('err_cap', 4)
                st.session_state[f"esa_{current_uid}"] = style.get('err_alpha', 0.2)
                st.session_state[f"ax_{current_uid}"] = style.get('axis', "Left")
                st.session_state[f"pst_{current_uid}"] = style.get('preset_color', "Blue")
                st.session_state[f"clr_{current_uid}"] = style.get('hex_color', "#0072BD")
                st.session_state[f"ls_{current_uid}"] = style.get('linestyle', "-")
                st.session_state[f"lw_{current_uid}"] = style.get('linewidth', 2.0)
                st.session_state[f"mk_{current_uid}"] = style.get('marker', "None")
                st.session_state[f"mf_{current_uid}"] = style.get('fill', "Solid")
                st.session_state[f"ms_{current_uid}"] = style.get('size', 8)
                st.session_state[f"sm_{current_uid}"] = style.get('smooth', False)
                st.session_state[f"sa_{current_uid}"] = style.get('smooth_algo', "Spline (Rounded)")
                st.session_state[f"tr_{current_uid}"] = style.get('trendline', "None")

                new_series.append({
                    "internal_id": current_uid,  # Permanent ID
                    "data_name": s['data_name'], "err_name": "Restored",
                    "df": df_main, "df_err": df_err, "x_col_name": s['x_col_name'], "y_col_name": s['y_col_name']
                })

            st.session_state.series_data = new_series
            st.session_state.next_id = max_id + 1  # Continue counting from here
            st.success("Project Loaded with Styles!")
        except Exception as e:
            st.error(f"Failed to load project: {e}")

    st.markdown("---")
    st.header("1. Add Data")
    data_files = st.file_uploader("Upload Data Files", type=["xlsx", "csv"], accept_multiple_files=True)
    err_files = st.file_uploader("Upload Error Files", type=["xlsx", "csv"], accept_multiple_files=True)

    data_file_dict = {f.name: f for f in data_files} if data_files else {}
    err_file_dict = {f.name: f for f in err_files} if err_files else {}

    if data_file_dict:
        c1, c2 = st.columns(2)
        selected_d_name = c1.selectbox("Select Data", list(data_file_dict.keys()))
        err_options = ["None"] + list(err_file_dict.keys())
        selected_e_name = c2.selectbox("Select Error", err_options)

        try:
            d_file = data_file_dict[selected_d_name];
            d_file.seek(0)
            if d_file.name.endswith('csv'):
                df_temp = pd.read_csv(d_file, nrows=0)
            else:
                df_temp = pd.read_excel(d_file, nrows=0)
            cols = df_temp.columns.tolist()
            def_x = cols.index('x') if 'x' in cols else 0
            x_col_select = st.selectbox("X Column", cols, index=def_x)
            y_cols_select = st.multiselect("Y Columns", [c for c in cols if c != x_col_select])

            if st.button(f"➕ Add Curves"):
                d_file.seek(0)
                if d_file.name.endswith('csv'):
                    df_d = pd.read_csv(d_file)
                else:
                    df_d = pd.read_excel(d_file)
                df_e = None
                if selected_e_name != "None":
                    e_file = err_file_dict[selected_e_name];
                    e_file.seek(0)
                    if e_file.name.endswith('csv'):
                        df_e = pd.read_csv(e_file)
                    else:
                        df_e = pd.read_excel(e_file)

                for y_c in y_cols_select:
                    # Assign a Unique ID and increment counter
                    uid = st.session_state.next_id
                    st.session_state.next_id += 1

                    st.session_state.series_data.append({
                        "internal_id": uid,  # Stores the unique key
                        "data_name": selected_d_name, "err_name": selected_e_name,
                        "df": df_d, "df_err": df_e, "x_col_name": x_col_select, "y_col_name": y_c
                    })
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.series_data:
        st.markdown("---")
        # Confirmation for Clear All
        if st.button("Clear All Data"):
            st.session_state.confirm_clear = True  # Set flag

        if st.session_state.get('confirm_clear', False):
            st.warning("Are you sure? This will delete all curves and settings.")
            col_yes, col_no = st.columns(2)
            if col_yes.button("Yes, delete everything", type="primary"):
                st.session_state.series_data = []
                st.session_state.confirm_clear = False
                st.rerun()
            if col_no.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

# --- MAIN AREA ---
if st.session_state.series_data:
    col_left, col_right = st.columns([1, 2])
    processed_series_list = []

    with col_left:
        st.subheader("Curve Styling")
        indices_to_remove = []

        # Iterate over series, BUT use the unique 'internal_id' for keys
        for i, series in enumerate(st.session_state.series_data):
            uid = series['internal_id']  # Use ID, not index 'i'

            df = series['df'];
            df_err = series['df_err'];
            y_col_name = series['y_col_name']

            with st.expander(f"#{i + 1}: {y_col_name}", expanded=False):
                # NOTE: All keys now use 'uid'
                custom_label = st.text_input("Label", value=y_col_name, key=f"lbl_{uid}")
                leg_order = st.number_input("Order", value=i, step=1, key=f"ord_{uid}")

                y_err_data = None;
                est = "Bar";
                esa = 0.2;
                ecs = 4
                if df_err is not None:
                    err_cols = ["None"] + df_err.columns.tolist()
                    def_err = 0
                    if y_col_name in df_err.columns: def_err = err_cols.index(y_col_name)
                    err_col_name = st.selectbox("Error Col", err_cols, index=def_err, key=f"ecn_{uid}")
                    if err_col_name != "None":
                        min_len = min(len(df), len(df_err))
                        y_err_data = df_err[err_col_name].iloc[:min_len].values
                        e1, e2 = st.columns(2)
                        est = e1.selectbox("Style", ["Bar", "Sleeve"], key=f"est_{uid}")
                        if est == "Bar":
                            ecs = e2.slider("Cap", 0, 10, 4, key=f"ecs_{uid}")
                        else:
                            esa = e2.slider("Alpha", 0.0, 1.0, 0.2, key=f"esa_{uid}")

                axis_side = st.radio("Axis", ["Left", "Right"], horizontal=True, key=f"ax_{uid}")
                axis_val = "Primary" if axis_side == "Left" else "Secondary"

                c1, c2 = st.columns(2)
                # Color sync with unique ID
                preset = c1.selectbox("Color", list(STANDARD_COLORS.keys()), index=i % len(STANDARD_COLORS),
                                      key=f"pst_{uid}", on_change=sync_color, args=(uid,))
                color = c2.color_picker("Hex", STANDARD_COLORS[preset], key=f"clr_{uid}")

                l1, l2 = st.columns(2)
                ls = l1.selectbox("Line", ["-", "--", "-.", ":", "None"], key=f"ls_{uid}")
                lw = l2.slider("Width", 0.5, 5.0, 2.0, key=f"lw_{uid}")

                m1, m2, m3 = st.columns(3)
                mk = m1.selectbox("Marker", ["None", "o", "s", "^", "D"], key=f"mk_{uid}")
                mf = m2.selectbox("Fill", ["Solid", "Hollow"], key=f"mf_{uid}")
                ms = m3.slider("Size", 1, 20, 8, key=f"ms_{uid}")

                sm_check = st.checkbox("Smooth", key=f"sm_{uid}")
                sm_algo = "Spline (Rounded)"
                if sm_check: sm_algo = st.selectbox("Algo", ["Spline (Rounded)", "Akima (Tight)", "PCHIP (Monotonic)"],
                                                    key=f"sa_{uid}")

                tr_type = st.selectbox("Trendline",
                                       ["None", "Linear", "Exponential", "Logarithmic", "Power", "Polynomial"],
                                       key=f"tr_{uid}")

                st.markdown("---")
                if st.button(f"🗑️ Remove", key=f"del_{uid}"): indices_to_remove.append(i)

                # Data Processing
                x_data = df[series['x_col_name']].values;
                y_data = df[y_col_name].values
                if y_err_data is not None:
                    min_l = min(len(x_data), len(y_err_data))
                    x_data = x_data[:min_l];
                    y_data = y_data[:min_l];
                    y_err_data = y_err_data[:min_l]

                processed_series_list.append({
                    "x_data": x_data, "y_data": y_data, "err_data": y_err_data,
                    "label": custom_label, "order": leg_order, "axis": axis_val,
                    "color": color, "linestyle": ls, "linewidth": lw, "marker": mk, "marker_fill": mf,
                    "marker_size": ms,
                    "smoothing": sm_check, "smooth_algo": sm_algo, "trendline": tr_type, "show_r2": True,
                    "error_style": est, "sleeve_alpha": esa, "capsize": ecs
                })

        if indices_to_remove:
            for idx in sorted(indices_to_remove, reverse=True): st.session_state.series_data.pop(idx)
            st.rerun()

    with col_right:
        st.subheader("Global Settings")
        with st.expander("Axes & Labels", expanded=True):
            t1, t2 = st.columns([3, 1])
            p_title = t1.text_input("Title", st.session_state.loaded_config.get("title", ""))
            fs_title = t2.number_input("Title Size", 1, None, st.session_state.loaded_config.get("fs_title", 18))

            x1, x2, x3 = st.columns([2, 2, 1])
            x_lab = x1.text_input("X Label", st.session_state.loaded_config.get("x_label", "x"))
            y_lab = x2.text_input("Y Label (Left)", st.session_state.loaded_config.get("y_label", "y"))
            fs_lab = x3.number_input("Lbl Size", 1, None, st.session_state.loaded_config.get("fs_label", 14))
            y_lab_r = st.text_input("Y Label (Right)", st.session_state.loaded_config.get("y_label_right", "y2"))

            tk1, tk2, tk3 = st.columns(3)
            fs_tick = tk1.number_input("Tick Size", 1, None, st.session_state.loaded_config.get("fs_ticks", 12))
            fs_leg = tk2.number_input("Leg Size", 1, None, st.session_state.loaded_config.get("fs_legend", 12))
            leg_cols = tk3.number_input("Leg Cols", 1, 10, st.session_state.loaded_config.get("legend_cols", 1))

            l_opts = ["best", "upper right", "upper left", "lower right", "None"]
            lpos = st.selectbox("Leg Pos", l_opts,
                                index=def_idx(st.session_state.loaded_config.get("legend_loc", "best"), l_opts))

            gr1, gr2 = st.columns(2)
            show_grid = gr1.checkbox("Grid", value=st.session_state.loaded_config.get("show_grid", False))
            leg_frame = gr2.checkbox("Leg Frame", value=st.session_state.loaded_config.get("legend_frame", False))

            l1, l2, l3 = st.columns(3)
            xlim_in = l1.text_input("X: min, step, max", list_to_str(st.session_state.loaded_config.get("x_lim", None)))
            ylim_in = l2.text_input("Y Left: min, step, max",
                                    list_to_str(st.session_state.loaded_config.get("y_lim", None)))
            ylim_r_in = l3.text_input("Y Right: min, step, max",
                                      list_to_str(st.session_state.loaded_config.get("y_lim_right", None)))

            xlim = str_to_list(xlim_in);
            ylim = str_to_list(ylim_in);
            ylim_right = str_to_list(ylim_r_in)
            mt1, mt2 = st.columns(2)
            mx = mt1.number_input("Min X", 0, 10, st.session_state.loaded_config.get("minor_ticks_x", 0))
            my = mt2.number_input("Min Y", 0, 10, st.session_state.loaded_config.get("minor_ticks_y", 0))

        plot_config = {
            "title": p_title, "fs_title": fs_title, "x_label": x_lab, "y_label": y_lab, "y_label_right": y_lab_r,
            "fs_label": fs_lab, "fs_ticks": fs_tick, "fs_legend": fs_leg,
            "show_grid": show_grid, "legend_loc": lpos, "legend_frame": leg_frame,
            "legend_cols": leg_cols, "x_lim": xlim, "y_lim": ylim, "y_lim_right": ylim_right,
            "minor_ticks_x": mx, "minor_ticks_y": my
        }

        proj_json = get_project_json()
        st.download_button("💾 Save Project (Data + Config)", proj_json, "my_project.json", "application/json",
                           type="primary")
        fig = create_advanced_plot(processed_series_list, plot_config)
        st.pyplot(fig, use_container_width=True)
        buf = BytesIO();
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download PNG", buf.getvalue(), "pro_plot.png", "image/png")
else:
    st.info("👈 Please upload files and select columns to plot.")
