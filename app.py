import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from io import BytesIO
from scipy.interpolate import make_interp_spline, Akima1DInterpolator, PchipInterpolator
from matplotlib.ticker import AutoMinorLocator

# --- Global Plotting Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
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
    x_new = np.linspace(x_s.min(), x_s.max(), 300)
    try:
        unique_x, indices = np.unique(x_s, return_index=True)
        unique_y = y_s[indices]
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
        return x_s, y_s
    except:
        return x_s, y_s


# --- Main Plotting Function ---
def create_advanced_plot(series_list, plot_settings):
    # Sort series based on user defined order before plotting
    # This affects both Z-order (layering) and Legend order
    series_list.sort(key=lambda x: x['order'])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Secondary Axis Logic
    ax2 = None
    has_secondary = any(s['axis'] == 'Secondary' for s in series_list)
    if has_secondary:
        ax2 = ax.twinx()

    lines = []
    labels = []

    for series in series_list:
        # 1. Extract Data
        x = series['x_data']
        y = series['y_data']
        y_err = series['err_data']

        # 2. Determine Axis
        current_ax = ax2 if (series['axis'] == 'Secondary' and ax2) else ax

        # 3. Styling
        color = series['color']
        label = series['label']
        ls = series['linestyle']
        lw = series['linewidth']
        mk = series['marker']
        ms = series['marker_size']
        mfc = 'white' if series['marker_fill'] == 'Hollow' else color
        mec = color

        # 4. Error Bars / Sleeve
        if y_err is not None:
            if series['error_style'] == "Sleeve":
                y_lower = y - y_err
                y_upper = y + y_err
                if series['smoothing']:
                    px_l, py_l = get_smooth_curve(x, y_lower, series['smooth_algo'])
                    px_u, py_u = get_smooth_curve(x, y_upper, series['smooth_algo'])
                    current_ax.fill_between(px_l, py_l, py_u, color=color, alpha=series['sleeve_alpha'], linewidth=0)
                else:
                    current_ax.fill_between(x, y_lower, y_upper, color=color, alpha=series['sleeve_alpha'], linewidth=0)
            else:
                current_ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor=color, elinewidth=1.5,
                                    capsize=series['capsize'])

        # 5. Main Line/Markers
        if series['smoothing']:
            px, py = get_smooth_curve(x, y, series['smooth_algo'])
            if ls != "None":
                current_ax.plot(px, py, color=color, linestyle=ls, linewidth=lw)
            if mk != "None":
                current_ax.plot(x, y, linestyle='None', marker=mk, markersize=ms,
                                markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2)
        else:
            final_ls = ls if ls != "None" else "none"
            final_mk = mk if mk != "None" else "none"
            current_ax.plot(x, y, color=color, linestyle=final_ls, linewidth=lw,
                            marker=final_mk, markersize=ms,
                            markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2)

        # 6. Trendline
        if series['trendline'] != "None":
            tx, ty, t_label = calculate_trendline(x, y, series['trendline'])
            if tx is not None:
                final_t_label = f"{t_label}" if series['show_r2'] else None
                current_ax.plot(tx, ty, color=color, linestyle=':', linewidth=1.5, alpha=0.8, label=final_t_label)

        # 7. Collect for Legend (Ghost Line)
        l, = current_ax.plot([], [], color=color, linestyle=ls if ls != "None" else "None", linewidth=lw,
                             marker=mk if mk != "None" else "None", markersize=ms,
                             markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2,
                             label=label)
        lines.append(l)
        labels.append(label)

    # --- Axes Configuration ---
    ax.set_xlabel(plot_settings['x_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
    ax.set_ylabel(plot_settings['y_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")

    if ax2 and plot_settings['y_label_right']:
        ax2.set_ylabel(plot_settings['y_label_right'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
        ax2.tick_params(axis='y', which='major', labelsize=plot_settings['fs_ticks'])
        if plot_settings['minor_ticks_y'] > 0:
            ax2.yaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_y'] + 1))
        # Set Y Limit for Secondary Axis
        if plot_settings['y_lim_right']:
            ax2.set_ylim(plot_settings['y_lim_right'])

    if plot_settings['title']:
        ax.set_title(plot_settings['title'], fontsize=plot_settings['fs_title'], fontname="Times New Roman", pad=15)

    ax.tick_params(axis='both', which='major', labelsize=plot_settings['fs_ticks'])

    if plot_settings['show_grid']:
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    if plot_settings['minor_ticks_x'] > 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_x'] + 1))
    if plot_settings['minor_ticks_y'] > 0:
        ax.yaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_y'] + 1))
    if plot_settings['minor_ticks_x'] > 0 or plot_settings['minor_ticks_y'] > 0:
        ax.tick_params(which='minor', direction='in', top=True, right=True)

    if plot_settings['x_lim']: ax.set_xlim(plot_settings['x_lim'])
    if plot_settings['y_lim']: ax.set_ylim(plot_settings['y_lim'])

    # --- Unified Legend ---
    if plot_settings['legend_loc'] != "None":
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])

        extra_h = []
        extra_l = []
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            if l not in labels and l is not None:
                extra_h.append(h)
                extra_l.append(l)

        final_handles = lines + extra_h
        final_labels = labels + extra_l

        ax.legend(final_handles, final_labels, loc=plot_settings['legend_loc'],
                  fontsize=plot_settings['fs_legend'], frameon=plot_settings['legend_frame'],
                  fancybox=False, edgecolor='black', ncol=plot_settings['legend_cols'])

    return fig


# --- Streamlit UI ---
st.set_page_config(page_title="Pro Plotter", layout="wide")
st.title("📈 Pro Plotter")

if 'series_data' not in st.session_state:
    st.session_state.series_data = []
if 'loaded_config' not in st.session_state:
    st.session_state.loaded_config = {}

with st.sidebar:
    st.header("⚙️ Configuration")
    conf_col1, conf_col2 = st.columns(2)
    uploaded_conf = conf_col1.file_uploader("Load Config", type=["json"], label_visibility="collapsed")
    if uploaded_conf:
        try:
            st.session_state.loaded_config = json.load(uploaded_conf)
            st.success("Loaded!")
        except:
            st.error("Invalid JSON")


    def get_conf(key, default):
        return st.session_state.loaded_config.get(key, default)


    st.markdown("---")
    st.header("1. Upload Files")
    data_files = st.file_uploader("Upload Data Files", type=["xlsx", "csv"], accept_multiple_files=True)
    err_files = st.file_uploader("Upload Error Files", type=["xlsx", "csv"], accept_multiple_files=True)

    data_file_dict = {f.name: f for f in data_files} if data_files else {}
    err_file_dict = {f.name: f for f in err_files} if err_files else {}

    st.markdown("---")
    st.header("2. Link & Add Curves")

    if data_file_dict:
        c1, c2 = st.columns(2)
        selected_d_name = c1.selectbox("Select Data File", list(data_file_dict.keys()))

        err_options = ["None"] + list(err_file_dict.keys())
        selected_e_name = c2.selectbox("Select Error File", err_options)

        try:
            d_file = data_file_dict[selected_d_name]
            d_file.seek(0)
            if d_file.name.endswith('csv'):
                df_temp = pd.read_csv(d_file, nrows=0)
            else:
                df_temp = pd.read_excel(d_file, nrows=0)
            cols = df_temp.columns.tolist()

            def_x = cols.index('x') if 'x' in cols else 0
            x_col_select = st.selectbox("Select X Column", cols, index=def_x)

            remaining_cols = [c for c in cols if c != x_col_select]
            y_cols_select = st.multiselect("Select Y Columns (Multiple)", remaining_cols)

            if st.button(f"➕ Add {len(y_cols_select)} Curves"):
                if not y_cols_select:
                    st.warning("Please select at least one Y column.")
                else:
                    d_file.seek(0)
                    if d_file.name.endswith('csv'):
                        df_d = pd.read_csv(d_file)
                    else:
                        df_d = pd.read_excel(d_file)

                    df_e = None
                    if selected_e_name != "None":
                        e_file = err_file_dict[selected_e_name]
                        e_file.seek(0)
                        if e_file.name.endswith('csv'):
                            df_e = pd.read_csv(e_file)
                        else:
                            df_e = pd.read_excel(e_file)

                    for y_c in y_cols_select:
                        st.session_state.series_data.append({
                            "id": len(st.session_state.series_data),
                            "data_name": selected_d_name,
                            "err_name": selected_e_name,
                            "df": df_d,
                            "df_err": df_e,
                            "x_col_name": x_col_select,
                            "y_col_name": y_c
                        })
                    st.success(f"Added {len(y_cols_select)} curves!")

        except Exception as e:
            st.error(f"Error reading headers: {e}")

    else:
        st.info("Upload files to start.")

    if st.session_state.series_data:
        st.markdown("---")
        st.subheader(f"Active Series ({len(st.session_state.series_data)})")
        if st.button("Clear All"):
            st.session_state.series_data = []
            st.rerun()

if st.session_state.series_data:
    col_left, col_right = st.columns([1, 2])

    processed_series_list = []

    with col_left:
        st.subheader("Curve Styling")

        for i, series in enumerate(st.session_state.series_data):
            df = series['df']
            df_err = series['df_err']
            y_col_name = series['y_col_name']

            with st.expander(f"#{i + 1}: {y_col_name} ({series['data_name']})", expanded=False):
                # Custom Label
                custom_label = st.text_input("Legend Label", value=y_col_name, key=f"lbl_{i}")

                # --- NEW: Legend Order ---
                # Default order is the index i
                leg_order = st.number_input("Order", value=i, step=1, key=f"ord_{i}",
                                            help="Lower numbers appear first in legend")

                # Error Matching
                y_err_data = None
                est = "Bar"
                esa = 0.2
                ecs = 4
                if df_err is not None:
                    err_cols = ["None"] + df_err.columns.tolist()
                    def_err = 0
                    if y_col_name in df_err.columns:
                        def_err = err_cols.index(y_col_name)
                    err_col_name = st.selectbox("Error Col", err_cols, index=def_err, key=f"ecn_{i}")

                    if err_col_name != "None":
                        min_len = min(len(df), len(df_err))
                        y_err_data = df_err[err_col_name].iloc[:min_len].values
                        e1, e2 = st.columns(2)
                        est = e1.selectbox("Err Style", ["Bar", "Sleeve"], key=f"est_{i}")
                        if est == "Bar":
                            ecs = e2.slider("Cap", 0, 10, 4, key=f"ecs_{i}")
                        else:
                            esa = e2.slider("Alpha", 0.0, 1.0, 0.2, key=f"esa_{i}")

                axis_side = st.radio("Axis", ["Left", "Right"], horizontal=True, key=f"ax_{i}")
                axis_val = "Primary" if axis_side == "Left" else "Secondary"

                c1, c2 = st.columns([1, 1])
                preset = c1.selectbox("Color", list(STANDARD_COLORS.keys()), index=i % len(STANDARD_COLORS),
                                      key=f"pst_{i}")
                color = c2.color_picker("Hex", STANDARD_COLORS[preset], key=f"clr_{i}")

                l1, l2 = st.columns(2)
                ls = l1.selectbox("Line", ["-", "--", "-.", ":", "None"], key=f"ls_{i}")
                lw = l2.slider("Width", 0.5, 5.0, 2.0, key=f"lw_{i}")

                m1, m2, m3 = st.columns(3)
                mk = m1.selectbox("Marker", ["None", "o", "s", "^", "D"], key=f"mk_{i}")
                mf = m2.selectbox("Fill", ["Solid", "Hollow"], key=f"mf_{i}")
                ms = m3.slider("Size", 1, 20, 8, key=f"ms_{i}")

                sm_check = st.checkbox("Smooth", key=f"sm_{i}")
                sm_algo = "Spline (Rounded)"
                if sm_check:
                    sm_algo = st.selectbox("Algo", ["Spline (Rounded)", "Akima (Tight)", "PCHIP (Monotonic)"],
                                           key=f"sa_{i}")

                tr_type = st.selectbox("Trendline",
                                       ["None", "Linear", "Exponential", "Logarithmic", "Power", "Polynomial"],
                                       key=f"tr_{i}")

                x_data = df[series['x_col_name']].values
                y_data = df[y_col_name].values
                if y_err_data is not None:
                    min_l = min(len(x_data), len(y_err_data))
                    x_data = x_data[:min_l]
                    y_data = y_data[:min_l]
                    y_err_data = y_err_data[:min_l]

                processed_series_list.append({
                    "x_data": x_data,
                    "y_data": y_data,
                    "err_data": y_err_data,
                    "label": custom_label,
                    "order": leg_order,  # Added order
                    "axis": axis_val,
                    "color": color,
                    "linestyle": ls,
                    "linewidth": lw,
                    "marker": mk,
                    "marker_fill": mf,
                    "marker_size": ms,
                    "smoothing": sm_check,
                    "smooth_algo": sm_algo,
                    "trendline": tr_type,
                    "show_r2": True,
                    "error_style": est,
                    "sleeve_alpha": esa,
                    "capsize": ecs
                })

    with col_right:
        st.subheader("Global Settings")
        with st.expander("Axes & Labels", expanded=True):
            t1, t2 = st.columns([3, 1])
            p_title = t1.text_input("Title", get_conf("title", ""))
            # Unlimited font size (min=1, max=None)
            fs_title = t2.number_input("Title Size", 1, None, get_conf("fs_title", 18))

            x1, x2, x3 = st.columns([2, 2, 1])
            x_lab = x1.text_input("X Label", get_conf("x_label", "x"))
            y_lab = x2.text_input("Y Label (Left)", get_conf("y_label", "y"))
            # Unlimited font size
            fs_lab = x3.number_input("Lbl Size", 1, None, get_conf("fs_label", 14))

            y_lab_r = ""
            if any(s['axis'] == 'Secondary' for s in processed_series_list):
                y_lab_r = st.text_input("Y Label (Right)", get_conf("y_label_right", "y2"))

            tk1, tk2, tk3 = st.columns(3)
            # Unlimited font size
            fs_tick = tk1.number_input("Tick Size", 1, None, get_conf("fs_ticks", 12))
            # Unlimited font size
            fs_leg = tk2.number_input("Leg Size", 1, None, get_conf("fs_legend", 12))

            leg_cols = tk3.number_input("Leg Cols", 1, 10, get_conf("legend_cols", 1))

            lpos = st.selectbox("Leg Pos", ["best", "upper right", "upper left", "lower right", "None"],
                                index=["best", "upper right", "upper left", "lower right", "None"].index(
                                    get_conf("legend_loc", "best")))

            gr1, gr2 = st.columns(2)
            show_grid = gr1.checkbox("Grid", value=get_conf("show_grid", False))
            leg_frame = gr2.checkbox("Leg Frame", value=get_conf("legend_frame", False))

            l1, l2, l3 = st.columns(3)


            def parse_lim(s): return [float(x) for x in s.split(',')] if ',' in s else None


            def lim_to_str(l): return ",".join(map(str, l)) if l else ""


            xlim_in = l1.text_input("X Lim", lim_to_str(get_conf("x_lim", None)))
            ylim_in = l2.text_input("Y Lim (Left)", lim_to_str(get_conf("y_lim", None)))

            # --- NEW: Right Y-Axis Limit ---
            ylim_r_in = l3.text_input("Y Lim (Right)", lim_to_str(get_conf("y_lim_right", None)))

            xlim = parse_lim(xlim_in)
            ylim = parse_lim(ylim_in)
            ylim_right = parse_lim(ylim_r_in)

            mt1, mt2 = st.columns(2)
            mx = mt1.number_input("Min X", 0, 10, get_conf("minor_ticks_x", 0))
            my = mt2.number_input("Min Y", 0, 10, get_conf("minor_ticks_y", 0))

        plot_config = {
            "title": p_title, "fs_title": fs_title,
            "x_label": x_lab, "y_label": y_lab, "y_label_right": y_lab_r,
            "fs_label": fs_lab, "fs_ticks": fs_tick, "fs_legend": fs_leg,
            "show_grid": show_grid, "legend_loc": lpos, "legend_frame": leg_frame,
            "legend_cols": leg_cols,
            "x_lim": xlim, "y_lim": ylim, "y_lim_right": ylim_right,  # Added Right Limit
            "minor_ticks_x": mx, "minor_ticks_y": my
        }

        conf_json = json.dumps(plot_config)
        st.download_button("💾 Save Settings", conf_json, "plot_config.json", "application/json")

        fig = create_advanced_plot(processed_series_list, plot_config)
        st.pyplot(fig, use_container_width=True)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download PNG", buf.getvalue(), "pro_plot.png", "image/png")

else:
    st.info("👈 Please upload files and select columns to plot.")
